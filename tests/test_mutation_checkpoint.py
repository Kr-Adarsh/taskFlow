"""Mutation checkpoints with scripted models and independent local state checks."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from test_real_acceptance import real_workspace, INVOICE, COMPLAINT
from test_v2_integration import act, plan
from backend.app.agent.provider import FakeProvider
from backend.app.agent.verifier import snapshot_state, state_delta
from backend.app.agent_v2.graph import AgentRunner
from backend.app.agent_v2.state import Verification
from backend.app.agent_v2.verifier import CapabilityVerifier
from backend.app.capabilities.registry import CapabilityRegistry
from backend.app.tools.base import ToolResult
from backend.app.workspace.db import get_db_connection
from backend.app.workspace.fault_injection import fault_manager

pytestmark = pytest.mark.anyio
HISTORY = Path('tests/fixtures/regressions/completed_ticket')
EVIDENCE = Path('docs/validation/post-mutation-checkpoint-20261006')


class ContractProvider(FakeProvider):
    def __init__(self, responses, schemas):
        super().__init__(responses)
        self.expected_schemas = list(schemas)
        self.schemas = []

    async def generate_structured(self, messages, schema, *args, **kwargs):
        self.schemas.append(schema.__name__)
        assert self.expected_schemas.pop(0) == schema.__name__
        return await super().generate_structured(messages, schema, *args, **kwargs)


def preserve(name, result, provider, events, before, **extra):
    EVIDENCE.mkdir(exist_ok=True)
    (EVIDENCE / name).write_text(json.dumps({
        'result': result, 'schemas': provider.schemas, 'events': events,
        'pre_state': before, 'post_state': snapshot_state(),
        'state_delta': state_delta(before, snapshot_state()), **extra,
    }, indent=2) + '\n')


async def test_latest_complaint_verifies_at_twelve_without_historical_decisions(real_workspace):
    record = json.loads((HISTORY / 'complaint.json').read_text())
    generations = record['generations']
    historical = [entry['response'] for entry in generations if entry['schema'] == 'Decision' and 'response' in entry]
    responses = [generations[0]['response'], *historical[:12],
                 {'collection': 'tickets', 'complaint_id': '4821', 'condition_tier': 'Enterprise', 'priority': 'High'},
                 {'accurate': True, 'reason': 'Summary matches the source complaint',
                  'source_quotes': ['severe database synchronization outages across our EU servers',
                                    'Our internal analytics have completely stalled.'], 'contradictions': []},
                 *historical[12:]]
    provider = ContractProvider(responses, ['TaskPlan'] + ['Decision'] * 12 + ['VerificationIntent', 'SummaryAssessment'])
    before, events = snapshot_state(), []
    result = await AgentRunner(provider=provider, event_callback=lambda _, kind, payload: events.append((kind, payload))).execute_task(record['objective'])
    assert result['status'] == 'completed' and result['steps'] == 12
    assert result['verification']['verified'] and not provider.expected_schemas
    assert provider.responses == historical[12:]
    assert [p['step'] for kind, p in events if kind == 'DECISION'] == list(range(1, 13))
    assert [p['step'] for kind, p in events if kind == 'VERIFICATION_CHECKPOINT'] == [12]
    assert all(p['decision']['action'] == 'act' for kind, p in events if kind == 'DECISION')
    names = [kind for kind, _ in events]
    checkpoint = names.index('VERIFICATION_CHECKPOINT')
    assert names[checkpoint + 1] == 'VERIFICATION_REQUESTED'
    assert not any(kind == 'ACTION' for kind, _ in events[checkpoint + 1:])
    criteria = result['verification']['criteria_results']
    crm = next(c for c in criteria if c['criterion'] == 'Source customer and exact CRM account resolved')
    assert crm['passed'] and crm['evidence']['customer'] == 'Acme Corp' and crm['evidence']['tier'] == 'Enterprise'
    assert next(c for c in criteria if c['criterion'] == 'Grounded complaint summary')['passed']
    audit = next(p for kind, p in events if kind == 'VERIFICATION' and p.get('stage') == 'original_objective')
    assert audit['verified']
    delta = state_delta(before, snapshot_state())
    row, = delta['tickets']['created']
    old = record['state_delta']['tickets']['created'][0]
    assert (row['ticket_id'], row['customer'], row['priority'], row['source_reference']) == ('TIK-8402', 'Acme Corp', 'High', 'complaint_4821.txt')
    assert row['summary'].splitlines() == old['summary'].splitlines()
    assert not delta['invoices']['created'] and not delta['accounts']['created']
    assert not any(changes[kind] for changes in delta.values() for kind in ('updated', 'deleted'))
    preserve('latest-complaint-replay.json', result, provider, events, before,
             avoided_historical_decisions=list(range(13, 18)), remaining_historical_responses=provider.responses)


async def test_invoice_precommit_503_then_checkpoint(real_workspace):
    fault_manager.arm('finance_create_invoice', 1)
    responses = [plan(INVOICE, 'browser'), act('read_document_chunks', document_id='acme_invoice_1044.pdf'),
                 act('browser_open', url='/workspace/finance')]
    for field, text in [('company', 'Acme Corp'), ('invoice_number', 'INV-1044'), ('amount', '84500'),
                        ('due_date', '2026-10-15'), ('source_reference', 'acme_invoice_1044.pdf')]:
        responses.append(act('browser_type', element_id='@' + field, text=text))
    responses.extend([act('browser_click', element_id='@submit_invoice'),
                      act('browser_click', element_id='@submit_invoice'),
                      {'collection': 'invoices', 'company': 'Acme Corp', 'selection': 'latest'}])
    provider = ContractProvider(responses, ['TaskPlan'] + ['Decision'] * 9 + ['VerificationIntent'])
    before, events = snapshot_state(), []
    result = await AgentRunner(provider=provider, event_callback=lambda _, kind, payload: events.append((kind, payload))).execute_task(INVOICE)
    assert result['status'] == 'completed' and result['steps'] == 9 and result['verification']['verified']
    assert not provider.responses and not provider.expected_schemas
    failed = next(p for kind, p in events if kind == 'OBSERVATION' and p.get('error_code') == 'TRANSIENT_503_ERROR')
    assert failed['step'] == 8 and failed['evidence']['commit_state'] == 'not_committed'
    assert failed['retriable'] and failed['evidence']['http_status'] == 503
    assert [p['step'] for kind, p in events if kind == 'VERIFICATION_CHECKPOINT'] == [9]
    assert [p['attempt'] for kind, p in events if kind == 'VERIFICATION_REQUESTED'] == [1]
    row, = state_delta(before, snapshot_state())['invoices']['created']
    assert (row['company'], row['invoice_number'], row['currency'], row['amount_minor'], row['due_date'], row['source_reference']) == ('Acme Corp', 'INV-1044', 'INR', 8450000, '2026-10-15', 'acme_invoice_1044.pdf')
    assert next(p for kind, p in events if kind == 'VERIFICATION' and p.get('stage') == 'original_objective')['verified']
    preserve('invoice-503-checkpoint.json', result, provider, events, before)


async def test_conditional_no_op_still_requires_model_ready(real_workspace):
    objective = COMPLAINT.replace('4821', '4822')
    responses = [plan(objective, 'browser'), act('read_document_chunks', document_id='complaint_4822.txt'),
                 act('browser_open', url='/workspace/crm'),
                 {'thought': 'Condition is false; request independent verification', 'action': 'ready_for_verification'},
                 {'collection': 'tickets', 'complaint_id': '4822', 'condition_tier': 'Enterprise', 'priority': 'High'}]
    provider = ContractProvider(responses, ['TaskPlan'] + ['Decision'] * 3 + ['VerificationIntent'])
    before, events = snapshot_state(), []
    result = await AgentRunner(provider=provider, event_callback=lambda _, kind, payload: events.append((kind, payload))).execute_task(objective)
    assert result['status'] == 'completed' and result['verification']['verified']
    assert not provider.responses and not any(kind == 'VERIFICATION_CHECKPOINT' for kind, _ in events)
    assert [p['decision']['action'] for kind, p in events if kind == 'DECISION'] == ['act', 'act', 'ready_for_verification']
    assert snapshot_state() == before
    assert next(c for c in result['verification']['criteria_results'] if c['criterion'] == 'Conditional outcome: no new or changed ticket')['passed']
    preserve('conditional-no-op.json', result, provider, events, before)


class AccountStateVerifier(CapabilityVerifier):
    """Test-only contract: independently require every requested MRR change."""
    def __init__(self, targets, fatal=False):
        self.targets = targets
        self.fatal = fatal
        self.checks = []

    async def verify(self, objective, task, before, memory, python_result=None):
        after = snapshot_state()
        self.checks.append({'state': after, 'guard_history': deepcopy(list(self.runner.oscillation_guard.history)),
                            'revision': self.runner.oscillation_guard.mutation_revision})
        actual = {row['id']: row['mrr'] for row in after['accounts']}
        missing = [f'Account {key} still requires MRR {value}' for key, value in self.targets.items() if actual[key] != value]
        outcome = 'FATAL_FAILURE' if self.fatal else 'RECOVERABLE_FAILURE' if missing else 'PASS'
        return Verification(outcome=outcome, verified=outcome == 'PASS', summary='Incomplete account changes' if missing else 'All requested changes independently checked',
                            discrepancies=missing, evidence={'state_delta': state_delta(before, after)})


def account_registry(tool_result, revisions):
    registry = CapabilityRegistry()
    async def persist_value(account_id, value):
        revisions.append(registry.runner.oscillation_guard.mutation_revision)
        with get_db_connection() as conn:
            conn.execute('UPDATE crm_accounts SET mrr=? WHERE id=?', (value, account_id))
            conn.commit()
        return tool_result
    registry.add('browser', 'browser_apply_value', 'Test-only account update',
                 {'type': 'object', 'properties': {'account_id': {'type': 'integer'}, 'value': {'type': 'integer'}},
                  'required': ['account_id', 'value']}, persist_value)
    return registry


@pytest.mark.parametrize('duplicate_receipt', [False, True])
async def test_incomplete_checkpoint_continues_and_second_mutation_verifies(real_workspace, duplicate_receipt):
    before, events, revisions = snapshot_state(), [], []
    first, second = before['accounts'][:2]
    targets = {first['id']: first['mrr'] + 1, second['id']: second['mrr'] + 2}
    result_data = {'outcome': 'confirmed_mutation', 'inspection': {'url': 'http://local.invalid/accounts',
                   'interactive_elements': [{'id': '@value', 'tag': 'input', 'value': ''}], 'tables': []}}
    registry = account_registry(ToolResult(ok=True, data=result_data), revisions)
    objective = 'Apply both requested account MRR changes'
    responses = [plan(objective, 'browser'), act('browser_apply_value', account_id=first['id'], value=targets[first['id']])]
    if duplicate_receipt:
        responses.append(act('browser_apply_value', account_id=first['id'], value=targets[first['id']]))
    responses.append(act('browser_apply_value', account_id=second['id'], value=targets[second['id']]))
    provider = ContractProvider(responses, ['TaskPlan'] + ['Decision'] * (3 if duplicate_receipt else 2))
    verifier = AccountStateVerifier(targets)
    runner = AgentRunner(provider=provider, tool_registry=registry, verifier=verifier,
                         event_callback=lambda _, kind, payload: events.append((kind, payload)))
    registry.runner = verifier.runner = runner
    result = await runner.execute_task(objective)
    assert result['status'] == 'completed' and result['verification']['verified']
    assert not provider.responses and len(verifier.checks) == 2
    checkpoints = [p['step'] for kind, p in events if kind == 'VERIFICATION_CHECKPOINT']
    assert checkpoints == [1, 3 if duplicate_receipt else 2]
    outcomes = [p['outcome'] for kind, p in events if kind == 'VERIFICATION' and 'task_id' in p]
    assert outcomes == ['RECOVERABLE_FAILURE', 'PASS']
    assert result['report']['tasks'][0]['verification_attempts'] == 2
    continuation = json.loads(provider.call_history[2][-1]['content'].split('\n', 1)[1])['untrusted_task_data']
    assert continuation['recent_outcomes']['observation']['verification_failed'] and continuation['recent_outcomes']['observation']['discrepancies']
    assert revisions == ([0, 1, 1] if duplicate_receipt else [0, 1])
    assert [check['revision'] for check in verifier.checks] == [1, 2]
    assert not runner.oscillation_guard.warned and runner.oscillation_guard.pending is None
    assert len(runner.oscillation_guard.history) == (3 if duplicate_receipt else 2)
    if duplicate_receipt:
        assert not list(runner.oscillation_guard.history)[1]['novel']
    assert result['report']['state_delta']['accounts']['updated'] and not result['report']['state_delta']['tickets']['created']
    preserve(f'multi-mutation-{duplicate_receipt}.json', result, provider, events, before, independent_checks=verifier.checks, revisions_at_dispatch=revisions)


@pytest.mark.parametrize('result,changed,checkpoint', [
    (ToolResult(ok=True, data={'outcome': 'confirmed_mutation'}), True, True),
    (ToolResult(ok=True, evidence={'commit_state': 'committed'}), True, True),
    (ToolResult(ok=True, data={'commit_state': 'committed'}), True, True),
    (ToolResult(ok=True, data={'outcome': 'confirmed_mutation'}), False, False),
    (ToolResult(ok=True, data={'outcome': 'confirmed_navigation'}), False, False),
    (ToolResult(ok=True, data={'outcome': 'confirmed_navigation'}), True, False),
    (ToolResult(ok=True, data={'value': 'locally edited form'}), False, False),
    (ToolResult(ok=False, retriable=True, evidence={'http_status': 503, 'commit_state': 'not_committed'}), False, False),
    (ToolResult(ok=False, error_code='UNCERTAIN_OUTCOME', evidence={'commit_state': 'unknown'}), False, False),
    (ToolResult(ok=True, data={'outcome': 'confirmed_mutation'}, evidence={'commit_state': 'unknown'}), True, False),
    (ToolResult(ok=True, data={'outcome': 'confirmed_mutation'}, evidence={'commit_state': 'not_committed'}), False, False),
    (ToolResult(ok=False, data={'outcome': 'confirmed_mutation'}), True, False),
])
async def test_only_authoritative_confirmed_persistent_change_checkpoints(real_workspace, result, changed, checkpoint):
    row = snapshot_state()['accounts'][0]
    target = row['mrr'] + int(changed)
    registry = account_registry(result, [])
    objective = 'Verify the requested account state'
    responses = [plan(objective, 'browser'), act('browser_apply_value', account_id=row['id'], value=target)]
    if not checkpoint:
        responses.append({'thought': 'Mutation prose is not a commit signal', 'action': 'ready_for_verification', 'result': {'claim': 'I changed business state'}})
    provider = ContractProvider(responses, ['TaskPlan'] + ['Decision'] * (1 if checkpoint else 2))
    verifier, events = AccountStateVerifier({row['id']: target}), []
    runner = AgentRunner(provider=provider, tool_registry=registry, verifier=verifier,
                         event_callback=lambda _, kind, payload: events.append((kind, payload)))
    registry.runner = verifier.runner = runner
    final = await runner.execute_task(objective)
    assert final['status'] == 'completed' and not provider.responses
    assert sum(kind == 'VERIFICATION_CHECKPOINT' for kind, _ in events) == int(checkpoint)
    assert final['report']['tasks'][0]['verification_attempts'] == 1


@pytest.mark.parametrize('fatal', [True, False])
async def test_checkpoint_rejection_keeps_fatal_and_attempt_limits(real_workspace, fatal):
    before = snapshot_state()
    targets = {row['id']: row['mrr'] + 1 for row in before['accounts']}
    registry = account_registry(ToolResult(ok=True, evidence={'commit_state': 'committed'}), [])
    objective = 'Update all requested accounts'
    actions = [act('browser_apply_value', account_id=row['id'], value=targets[row['id']]) for row in before['accounts']]
    provider = ContractProvider([plan(objective, 'browser'), *actions], ['TaskPlan'] + ['Decision'] * (1 if fatal else 2))
    verifier = AccountStateVerifier(targets, fatal=fatal)
    runner = AgentRunner(provider=provider, tool_registry=registry, verifier=verifier)
    registry.runner = verifier.runner = runner
    final = await runner.execute_task(objective)
    assert final['status'] == 'failed'
    assert final['report']['tasks'][0]['verification_attempts'] == (1 if fatal else 2)
    assert len(provider.responses) == (2 if fatal else 1)
    assert runner.max_steps == 20 and runner.max_verification_attempts == 2 and runner.run_deadline_seconds == 600


@pytest.mark.parametrize('capability', ['documents', 'python'])
async def test_checkpoint_cannot_bypass_read_only_verifier_semantics(real_workspace, capability):
    row = snapshot_state()['accounts'][0]
    registry = account_registry(ToolResult(ok=True, data={'outcome': 'confirmed_mutation'}), [])
    objective = 'Produce a read-only analysis'
    provider = ContractProvider([plan(objective, capability), act('browser_apply_value', account_id=row['id'], value=row['mrr'] + 1)],
                                ['TaskPlan', 'Decision'])
    runner = AgentRunner(provider=provider, tool_registry=registry)
    registry.runner = runner
    result = await runner.execute_task(objective)
    assert result['status'] == 'failed' and not provider.responses
    assert result['report']['verification']['outcome'] == 'FATAL_FAILURE'
    assert 'read-only' in result['error']
    assert result['report']['tasks'][0]['verification_attempts'] == 1
