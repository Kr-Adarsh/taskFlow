"""Offline context ownership checks using preserved observations and local browser."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from test_v2_progress import browser_workspace
from test_real_acceptance import real_workspace
from backend.app.agent_v2.context import ContextMemory, decision_prompt
from backend.app.agent_v2.state import GraphState, Subtask
from backend.app.capabilities.documents import read_document_chunks
from backend.app.capabilities.python.profile import dataset_profile
from backend.app.tools.base import ToolResult
from backend.app.tools.browser_tools import browser_manager

HISTORY = Path('tests/fixtures/regressions/browser_context')
EVIDENCE = Path('docs/validation/current-browser-context-20261006')


def model_data(messages):
    return json.loads(messages[-1]['content'].split('\n', 1)[1])['untrusted_task_data']


def selectors(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ('id', 'element', 'element_id', 'selector', 'locator') and isinstance(item, str) and item.startswith('@'):
                yield item
            yield from selectors(item)
    elif isinstance(value, list):
        for item in value:
            yield from selectors(item)


def prompt(memory, observation=None, objective='Complete the requested outcome'):
    task = Subtask(task_id='task', goal=objective, verification_capability='browser', success_criteria=[objective])
    state = GraphState(run_id='offline', objective=objective, tasks=[task], current_task_id='task',
                       observation=observation or {})
    return decision_prompt(state, task, memory, [])


def replay(record, until, memory=None):
    memory = memory or ContextMemory()
    actions = {event['payload']['step']: event['payload'] for event in record['events'] if event['event_type'] == 'ACTION'}
    last = {}
    for event in record['events']:
        if event['event_type'] != 'OBSERVATION' or event['payload']['step'] > until:
            continue
        payload = event['payload']
        action = actions[payload['step']]
        last = {key: payload[key] for key in ToolResult.model_fields if key in payload}
        memory.update(action['tool'], action['args'], ToolResult.model_validate(last))
    return memory, last


@pytest.mark.parametrize('decision,unavailable', [(7, '@ticket_priority'), (11, '@ticket_customer')])
def test_exact_failed_context_has_only_current_crm_controls(decision, unavailable):
    record = json.loads((HISTORY / 'complaint.json').read_text())
    memory, observation = replay(record, decision - 1)
    messages = prompt(memory, observation, record['objective'])
    data = model_data(messages)
    current = data['current_browser_state']
    assert current['url'].endswith('/workspace/crm')
    current_ids = set(selectors(current))
    assert '@search_crm' in current_ids and '@button_search_crm' in current_ids
    assert unavailable not in current_ids
    other = {key: value for key, value in data.items() if key != 'current_browser_state'}
    assert not list(selectors(other))
    assert unavailable not in json.dumps(data)
    support = next(page for page in data['historical_evidence'] if page['source'].endswith('/workspace/support'))
    assert support['observed_values'] and any(value.get('label') == 'Customer Name' for value in support['observed_values'])
    original = json.loads((HISTORY / 'model-visible-context.json').read_text())[decision - 1]['actual_serialized_task_data']
    assert unavailable in json.dumps(original['memory']['pages'])
    (EVIDENCE / f'decision-{decision}-counterfactual.json').write_text(json.dumps({
        'original_serialized_task_data': original, 'new_messages': messages,
        'new_task_data': data, 'current_ids': sorted(current_ids), 'historical_selector_removed': unavailable,
    }, indent=2) + '\n')


def test_table_facts_are_established_with_provenance_after_navigation():
    record = json.loads((HISTORY / 'complaint.json').read_text())
    memory, observation = replay(record, 4)
    data = model_data(prompt(memory, observation, record['objective']))
    assert data['current_browser_state']['url'].endswith('/workspace/support')
    evidence = next(item for item in data['established_evidence'] if item['source'].endswith('/workspace/crm'))
    assert evidence['kind'] == 'browser_tables' and evidence['durable'] is True
    assert any('Acme Corp' in row and 'Enterprise' in row for table in evidence['facts']['tables'] for row in table[1:])
    assert 'is_enterprise' not in evidence['facts']
    assert not list(selectors(evidence))
    source = next(item for item in data['established_evidence'] if item['kind'] == 'browser_preview')
    assert source['source'].endswith('view=complaint_4821.txt') and 'Customer Name: Acme Corp' in source['facts']['text']
    (EVIDENCE / 'established-source-evidence.json').write_text(json.dumps(data, indent=2) + '\n')


@pytest.mark.anyio
async def test_finance_controls_survive_document_read_and_become_history_on_navigation(browser_workspace):
    memory = ContextMemory()
    opened = await browser_manager.browser_open('/workspace/finance')
    memory.update('browser_open', {'url': '/workspace/finance'}, opened)
    typed = await browser_manager.browser_type('@company', 'Acme Corp')
    memory.update('browser_type', {'element_id': '@company', 'text': 'Acme Corp'}, typed)
    current = deepcopy(memory.current_browser_state)
    read = read_document_chunks('acme_invoice_1044.pdf')
    memory.update('read_document_chunks', {'document_id': 'acme_invoice_1044.pdf'}, read)
    after_read = model_data(prompt(memory, read.model_dump()))
    assert after_read['current_browser_state'] == current
    controls = {item['id']: item for item in current['interactive_elements']}
    assert controls['@company']['value'] == 'Acme Corp'
    assert controls['@source_reference']['required'] and '@submit_invoice' in controls
    assert any(item['kind'] == 'document_chunks' for item in after_read['established_evidence'])
    opened = await browser_manager.browser_open('/workspace/crm')
    memory.update('browser_open', {'url': '/workspace/crm'}, opened)
    after_navigation = model_data(prompt(memory, opened.model_dump()))
    assert after_navigation['current_browser_state']['url'].endswith('/workspace/crm')
    assert '@company' not in set(selectors(after_navigation)) and '@submit_invoice' not in set(selectors(after_navigation))
    finance = next(item for item in after_navigation['historical_evidence'] if item['source'].endswith('/workspace/finance'))
    assert any(item.get('value') == 'Acme Corp' for item in finance['observed_values'])
    assert '@submit_invoice' in json.dumps(memory.get_snapshot()['pages'])
    (EVIDENCE / 'invoice-continuity.json').write_text(json.dumps({
        'after_document_read': after_read, 'after_navigation': after_navigation,
        'raw_runtime_snapshot': memory.get_snapshot(),
    }, indent=2) + '\n')


def test_raw_guard_and_verifier_snapshots_match_preserved_implementation():
    namespace = {'__name__': 'preserved_context'}
    exec(compile((HISTORY / 'context_before.py').read_text(), 'preserved_context', 'exec'), namespace)
    old, new = namespace['ContextMemory'](), ContextMemory()
    record = json.loads((HISTORY / 'complaint.json').read_text())
    actions = {e['payload']['step']: e['payload'] for e in record['events'] if e['event_type'] == 'ACTION'}
    for event in record['events']:
        if event['event_type'] != 'OBSERVATION':
            continue
        payload = event['payload']
        if payload['step'] not in actions:
            continue
        action = actions[payload['step']]
        result = ToolResult.model_validate({key: payload[key] for key in ToolResult.model_fields if key in payload})
        for memory in (old, new):
            memory.update(action['tool'], action['args'], result)
        assert old.get_snapshot() == new.get_snapshot()
        before = deepcopy(new.get_snapshot())
        prompt(new, result.model_dump())
        assert new.get_snapshot() == before


def inspected(url, value='', tables=None, options=None):
    return ToolResult(ok=True, data={'inspection': {'url': url, 'title': 'Workspace', 'tables': tables or [],
        'interactive_elements': [{'id': '@entry', 'tag': 'select', 'type': 'select-one', 'label': 'Entry',
                                  'value': value, 'required': True, 'disabled': False, 'options': options or []}]}})


def test_non_browser_inspection_cannot_take_over_current_state():
    memory = ContextMemory()
    memory.update('browser_inspect', {}, inspected('/current', 'chosen'))
    current = deepcopy(memory.current_browser_state)
    other = inspected('/not-current', 'other')
    memory.update('inspect_file', {'document_id': 'notes.txt'}, other)
    data = model_data(prompt(memory, other.model_dump()))
    assert data['current_browser_state'] == current and not list(selectors(data['recent_outcomes']))


def test_failed_browser_observation_updates_actual_page_not_requested_target():
    memory = ContextMemory()
    memory.update('browser_inspect', {}, inspected('/original'))
    failure = inspected('/actual-after-submit', 'observed')
    failure.ok = False
    failure.error_code = 'TRANSIENT_503_ERROR'
    failure.retriable = True
    memory.update('browser_click', {'element_id': '@send'}, failure)
    assert memory.current_browser_state['url'] == '/actual-after-submit'
    assert memory.current_browser_state['interactive_elements'][0]['value'] == 'observed'
    missing = ToolResult(ok=False, error_code='UNCERTAIN_OUTCOME')
    memory.update('browser_open', {'url': '/imagined'}, missing)
    assert memory.current_browser_state['url'] == '/actual-after-submit'


@pytest.mark.parametrize('arguments', [{'element_id': '@entry', 'text': 'value'}, {'@entry': 'value'}])
def test_recent_outcomes_and_guard_feedback_do_not_reintroduce_historical_selectors(arguments):
    memory = ContextMemory()
    memory.update('browser_inspect', {}, inspected('/previous', 'value'))
    memory.update('browser_type', arguments, inspected('/previous', 'value'))
    now = inspected('/current')
    now.data['inspection']['interactive_elements'][0]['id'] = '@current_entry'
    memory.update('browser_open', {'url': '/current'}, now)
    warning = ToolResult(ok=False, error_code='OSCILLATION_DETECTED', data={
        'cycle_length': 2, 'pattern': [{'page': '/previous', 'tool': 'browser_type', 'args': {'element_id': '@entry', 'text': 'value'}}]})
    original = deepcopy(warning.model_dump())
    data = model_data(prompt(memory, warning.model_dump()))
    assert list(selectors(data)) == ['@current_entry']
    assert '@entry' not in json.dumps(data)
    assert data['recent_outcomes']['observation']['error_code'] == 'OSCILLATION_DETECTED'
    assert warning.model_dump() == original


def test_table_evidence_survives_dom_eviction_and_replaces_changed_observations():
    memory = ContextMemory()
    for number in range(10):
        memory.update('browser_inspect', {}, inspected(f'/page/{number}', tables=[[['Name', 'Class'], [str(number), 'Observed']]]))
    assert len(memory.pages) == 3 and len(memory.table_evidence) == 8
    data = memory.relevant()
    assert any(item['source'] == '/page/2' for item in data['established_evidence'])
    assert '/page/2' not in memory.get_snapshot()['pages']
    memory.update('browser_inspect', {}, inspected('/page/2', tables=[[['Name', 'Class'], ['2', 'Changed']]]))
    evidence = [item for item in memory.relevant()['established_evidence'] if item['source'] == '/page/2']
    assert len(evidence) == 1 and evidence[0]['facts']['tables'][0][1][1] == 'Changed'


def test_current_controls_remain_compact_and_view_cannot_mutate_memory():
    memory = ContextMemory()
    options = [{'value': str(i), 'label': f'Choice {i}'} for i in range(30)]
    memory.update('browser_inspect', {}, inspected('/current', '29', options=options))
    data = memory.relevant()
    field = data['current_browser_state']['interactive_elements'][0]
    assert field['required'] and len(field['options']) == 16 and field['options_truncated']
    assert field['selected_option'] == {'value': '29', 'label': 'Choice 29'}
    field['value'] = 'tampered'
    assert memory.current_browser_state['interactive_elements'][0]['value'] == '29'


def test_visible_document_chunks_are_preserved_once_as_established_evidence():
    memory = ContextMemory()
    chunks = [{'document_id': 'notes.txt', 'chunk_id': f'notes.txt:1:{i}', 'page': 1, 'section': 'Notes', 'text': f'Observed fact {i}'} for i in range(3)]
    result = ToolResult(ok=True, data={'document_id': 'notes.txt', 'chunks': chunks, 'returned_chunks': 3})
    memory.update('read_document_chunks', {'document_id': 'notes.txt'}, result)
    data = model_data(prompt(memory, result.model_dump()))
    assert data['current_browser_state'] is None
    evidence = next(item for item in data['established_evidence'] if item['kind'] == 'document_chunks')
    assert evidence['facts']['chunks'] == chunks
    assert 'chunks' not in data['recent_outcomes']['observation']['data']
    assert result.data['chunks'] == chunks


def test_dataset_context_keeps_bounded_profile_without_raw_csv():
    memory = ContextMemory()
    result = ToolResult(ok=True, data=dataset_profile('sales.csv'))
    memory.update('profile_dataset', {'document_id': 'sales.csv'}, result)
    data = model_data(prompt(memory, result.model_dump(), 'Analyze revenue changes in sales.csv'))
    profile = next(item for item in data['established_evidence'] if item['kind'] == 'dataset_profile')
    assert profile['facts']['shape'][0] == 800 and len(profile['facts']['sample_rows']) == 3
    assert data['recent_outcomes']['observation']['data']['profile_available_in'] == 'established_evidence'
    assert data['current_browser_state'] is None
