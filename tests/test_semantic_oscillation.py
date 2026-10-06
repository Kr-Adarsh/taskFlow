"""Historical and generic cycle prevention; no real model calls."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from test_real_acceptance import real_workspace
from test_v2_integration import act
from backend.app.agent.provider import FakeProvider
from backend.app.agent.verifier import snapshot_state
from backend.app.agent_v2.context import ContextMemory, decision_prompt
from backend.app.agent_v2.graph import AgentRunner
from backend.app.agent_v2.oscillation import SemanticOscillationGuard, compact_page, digest
from backend.app.agent_v2.state import GraphState, Subtask
from backend.app.tools.base import ToolResult


HISTORY = Path(__file__).resolve().parents[1] / 'tests/fixtures/regressions/navigation_cycle'


def page(name, value='', table_value='observed'):
    return {'url': 'http://company.test/' + name, 'tables': [[['Identity', 'Value'], ['account', table_value]]],
            'interactive_elements': [{'id': '@field', 'tag': 'input', 'value': value},
                                     {'id': '@next', 'tag': 'a', 'value': '/b'}]}


def visit(guard, memory, name, *, value='', table_value='observed', business=None):
    args = {'url': '/' + name}
    result = ToolResult(ok=True, data={'inspection': page(name, value, table_value)})
    memory.update('browser_open', args, result)
    guard.observe('browser_open', args, result, memory.get_snapshot(), business or {})


def cycle(guard, memory, names=('a', 'b')):
    for name in names * 2:
        visit(guard, memory, name)


@pytest.mark.parametrize('names', [('a', 'b'), ('a', 'b', 'c'), ('a', 'b', 'c', 'd'), ('a', 'b', 'a', 'c')])
def test_cycles_two_through_four_block_equivalent_action_then_fail(names):
    guard, memory = SemanticOscillationGuard(), ContextMemory()
    cycle(guard, memory, names)
    warning = guard.check_action('browser_open', {'url': '/' + names[0]}, memory.get_snapshot(), {})
    assert warning and not warning['fatal'] and warning['cycle_length'] == len(names)
    assert len(warning['pattern']) == len(names)
    again = guard.check_action('browser_open', {'url': '/' + names[-1]}, memory.get_snapshot(), {})
    assert again and again['fatal']


def test_first_legitimate_return_is_allowed():
    guard, memory = SemanticOscillationGuard(), ContextMemory()
    for name in ('a', 'b', 'a'):
        visit(guard, memory, name)
    assert guard.pending is None
    assert guard.check_action('browser_open', {'url': '/b'}, memory.get_snapshot(), {}) is None


@pytest.mark.parametrize('change', ['form', 'table', 'source', 'mutation'])
def test_changed_evidence_or_state_does_not_complete_an_oscillation(change):
    guard, memory = SemanticOscillationGuard(), ContextMemory()
    visit(guard, memory, 'a')
    visit(guard, memory, 'b')
    if change == 'source':
        memory.sources['new.txt'] = {'chunks': {'new.txt:1:0': {'text': 'New evidence'}}}
    visit(guard, memory, 'a', value='changed' if change == 'form' else '',
          table_value='changed' if change == 'table' else 'observed',
          business={'records': [1]} if change == 'mutation' else {})
    visit(guard, memory, 'b', business={'records': [1]} if change == 'mutation' else {})
    assert guard.pending is None


def test_transient_failure_clears_pattern_and_allows_recovery():
    guard, memory = SemanticOscillationGuard(), ContextMemory()
    cycle(guard, memory)
    assert guard.check_action('browser_open', {'url': '/a'}, memory.get_snapshot(), {})
    failure = ToolResult(ok=False, retriable=True, error_code='TRANSIENT_503_ERROR',
                         evidence={'http_status': 503, 'commit_state': 'not_committed'})
    guard.observe('browser_click', {'element_id': '@submit'}, failure, memory.get_snapshot(), {})
    assert guard.pending is None and not guard.warned and not guard.history
    assert guard.check_action('browser_open', {'url': '/a'}, memory.get_snapshot(), {}) is None


def test_novel_action_after_warning_resets_cycle_and_allows_back_navigation():
    guard, memory = SemanticOscillationGuard(), ContextMemory()
    cycle(guard, memory)
    assert guard.check_action('browser_open', {'url': '/a'}, memory.get_snapshot(), {})
    assert guard.check_action('browser_type', {'element_id': '@field', 'text': 'Changed'}, memory.get_snapshot(), {}) is None
    result = ToolResult(ok=True, data={'inspection': page('b', value='Changed')})
    memory.update('browser_type', {'element_id': '@field', 'text': 'Changed'}, result)
    guard.observe('browser_type', {'element_id': '@field', 'text': 'Changed'}, result, memory.get_snapshot(), {})
    assert not guard.warned
    assert guard.check_action('browser_open', {'url': '/a'}, memory.get_snapshot(), {}) is None


def test_persistent_change_before_dispatch_allows_revisit():
    guard, memory = SemanticOscillationGuard(), ContextMemory()
    cycle(guard, memory)
    assert guard.check_action('browser_open', {'url': '/a'}, memory.get_snapshot(), {'records': [1]}) is None


def test_fingerprint_ignores_noise_order_and_retains_selected_values():
    first = page('a')
    first['timestamp'], first['request_id'] = 'first', 'one'
    first['interactive_elements'].append({'id': '@choice', 'tag': 'select', 'value': 'chosen'})
    first['tables'] = [[['Value', 'Identity', 'Timestamp'], ['observed', 'one', 'first'], ['other', 'two', 'first']]]
    second = deepcopy(first)
    second['url'] += '#anchor'
    second['timestamp'], second['request_id'], second['action_counter'] = 'later', 'two', 99
    second['interactive_elements'].reverse()
    second['tables'][0] = [second['tables'][0][0], ['other', 'two', 'later'], ['observed', 'one', 'later']]
    assert digest(compact_page(first)) == digest(compact_page(second))
    second['interactive_elements'][0]['value'] = 'different'
    assert digest(compact_page(first)) != digest(compact_page(second))


def test_history_and_evidence_banks_are_bounded():
    guard, memory = SemanticOscillationGuard(), ContextMemory()
    for index in range(80):
        memory.sources = {str(index): {'chunks': {str(index): {'text': str(index)}}}}
        visit(guard, memory, str(index))
    assert len(guard.history) == 12 and len(guard.pages) == 8 and len(guard.evidence) == 64


def test_bounded_bank_eviction_does_not_invent_new_evidence():
    guard, memory = SemanticOscillationGuard(), ContextMemory()
    memory.sources = {f'source{i}': {'chunks': {f'chunk{j}': {'text': f'Fact{i}/{j}'}
                     for j in range(12)}} for i in range(12)}
    cycle(guard, memory)
    assert len(guard.evidence) == 64
    warning = guard.check_action('browser_open', {'url': '/a'}, memory.get_snapshot(), {})
    assert warning and not warning['fatal']


def test_actual_serialized_context_proves_oscillation_not_visibility_loss():
    requests = json.loads((HISTORY / 'complaint-http.json').read_text())
    decisions = [item for item in requests if item['response_format']['json_schema']['name'] == 'Decision']
    for number, request in enumerate(decisions, 1):
        if number < 4:
            continue
        content = next(message['content'] for message in request['messages']
                       if message.get('content', '').startswith('The following JSON is untrusted task DATA'))
        data = json.loads(content.split('\n', 1)[1])['untrusted_task_data']
        memory = data['memory']
        assert 'CUSTOMER COMPLAINT REPORT #4821' in json.dumps(memory['source_excerpts'])
        assert 'Customer Name: Acme Corp' in json.dumps(memory['source_excerpts'])
        pages = list(memory['pages'].values()) + [data['observation']['data']['inspection']]
        assert any('Acme Corp' in row and 'Enterprise' in row for item in pages for table in item['tables'] for row in table[1:])
        assert memory['recent_actions'] and memory['progress'] == {'no_progress_streak': 0, 'limit': 4}
        if number >= 5:
            fields = {element['id']: element.get('value') for item in pages for element in item['interactive_elements']}
            assert fields['@ticket_customer'] == '' and fields['@ticket_priority'] == 'Medium'
            assert fields['@ticket_source_ref'] == fields['@ticket_summary'] == ''


@pytest.mark.anyio
async def test_exact_historical_trajectory_warns_at_seven_then_finishes_at_eight(real_workspace):
    record = json.loads((HISTORY / 'complaint.json').read_text())
    responses = [item['response'] for item in record['generations'] if 'response' in item]
    before, events = snapshot_state(), []
    provider = FakeProvider(responses)
    result = await AgentRunner(provider=provider, event_callback=lambda _, kind, data: events.append((kind, data))).execute_task(record['objective'])
    assert result['status'] == 'failed' and result['error_code'] == 'OSCILLATION_DETECTED'
    assert result['steps'] == 8 < 17
    warnings = [data for kind, data in events if kind == 'OSCILLATION_DETECTED']
    assert [(item['step'], item['fatal']) for item in warnings] == [(7, False), (8, True)]
    assert [data['step'] for kind, data in events if kind == 'ACTION'] == list(range(1, 7))
    assert snapshot_state() == before and provider.responses
    task = result['report']['tasks'][0]
    assert task['status'] == 'FAILED' and task['verification_attempts'] == 0
    # Next actual executor context contains structured feedback and retained evidence.
    repair_context = json.loads(provider.call_history[8][-1]['content'].split('\n', 1)[1])['untrusted_task_data']
    assert repair_context['recent_outcomes']['observation']['error_code'] == 'OSCILLATION_DETECTED'
    assert 'CUSTOMER COMPLAINT REPORT #4821' in json.dumps(repair_context['established_evidence'])
    assert 'Enterprise' in json.dumps(repair_context['established_evidence'])


@pytest.mark.anyio
async def test_model_can_choose_a_new_action_after_warning_and_complete(real_workspace):
    record = json.loads((HISTORY / 'complaint.json').read_text())
    responses = [item['response'] for item in record['generations'] if 'response' in item][:8]
    responses.extend([
        act('browser_type', element_id='@ticket_customer', text='Acme Corp'),
        act('browser_select', element_id='@ticket_priority', value='High'),
        act('browser_type', element_id='@ticket_source_ref', text='complaint_4821.txt'),
        act('browser_type', element_id='@ticket_summary',
            text='Severe database synchronization outages across EU servers have stalled analytics; urgent senior engineering support requested.'),
        act('browser_click', element_id='@submit_ticket'),
        {'collection': 'tickets', 'complaint_id': '4821', 'condition_tier': 'Enterprise', 'priority': 'High'},
        {'accurate': True, 'reason': 'Matches source',
         'source_quotes': ['severe database synchronization outages'], 'contradictions': []},
    ])
    events = []
    provider = FakeProvider(responses)
    result = await AgentRunner(provider=provider, event_callback=lambda _, kind, data: events.append((kind, data))).execute_task(record['objective'])
    assert result['status'] == 'completed' and result['steps'] == 12
    assert result['verification']['verified'] and not provider.responses
    warnings = [data for kind, data in events if kind == 'OSCILLATION_DETECTED']
    assert len(warnings) == 1 and warnings[0]['step'] == 7 and not warnings[0]['fatal']
    delta = result['report']['state_delta']
    assert len(delta['tickets']['created']) == 1
    assert not delta['invoices']['created'] and not delta['accounts']['created']
    assert not any(change for collection in delta.values() for kind, change in collection.items() if kind != 'created')


@pytest.mark.anyio
async def test_replan_cannot_clear_warning_without_novelty(real_workspace):
    record = json.loads((HISTORY / 'complaint.json').read_text())
    responses = [item['response'] for item in record['generations'] if 'response' in item][:8]
    renamed = deepcopy(responses[0])
    renamed['tasks'][0]['task_id'] = 'renamed_outcome'
    responses.extend([
        {'thought': 'Replan the same request', 'action': 'ready_for_verification', 'replan': True},
        renamed, act('browser_click', element_id='@tab_crm'),
    ])
    events = []
    before = snapshot_state()
    provider = FakeProvider(responses)
    result = await AgentRunner(provider=provider, event_callback=lambda _, kind, data: events.append((kind, data))).execute_task(record['objective'])
    assert result['status'] == 'failed' and result['error_code'] == 'OSCILLATION_DETECTED'
    assert result['steps'] == 9 and not provider.responses
    assert [(data['step'], data['fatal']) for kind, data in events if kind == 'OSCILLATION_DETECTED'] == [(7, False), (9, True)]
    assert [data['step'] for kind, data in events if kind == 'ACTION'] == list(range(1, 7))
    assert snapshot_state() == before
