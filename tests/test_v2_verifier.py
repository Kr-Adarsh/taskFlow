"""Verification contracts, state audits and model-call boundaries."""
import pytest

from backend.app.agent.provider import FakeProvider
from backend.app.agent.schemas import VerificationIntent
from backend.app.agent.verifier import snapshot_state, state_delta
from backend.app.agent_v2.context import ContextMemory
from backend.app.agent_v2.state import Subtask, SourceReference
from backend.app.agent_v2.verifier import CapabilityVerifier
from backend.app.capabilities.documents import chunks
from backend.app.workspace.db import get_db_connection
from backend.app.workspace.models import InvoiceCreate
from backend.app.workspace.seed import reset_demo_env
from backend.app.workspace.service import create_invoice

pytestmark = pytest.mark.anyio


class ContractProvider(FakeProvider):
    def __init__(self, intent):
        super().__init__([intent])
        self.schemas = []

    async def generate_structured(self, messages, response_schema, temperature=0.0):
        self.schemas.append(response_schema)
        assert issubclass(response_schema, VerificationIntent), 'Unexpected routing or coverage model call'
        return await super().generate_structured(messages, response_schema, temperature)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv('TASKFLOW_DB_PATH', str(tmp_path / 'verify.db'))
    reset_demo_env()


def persist_invoice():
    return create_invoice(InvoiceCreate(company='Acme Corp', invoice_number='INV-1044',
        amount='84500', currency='INR', due_date='2026-10-15', source_reference='acme_invoice_1044.pdf'))


def account_task(result):
    return Subtask(task_id='account', goal='Read CRM account', success_criteria=['Requested fields match CRM'],
                   verification_capability='browser', result=result)


def account_contract(fields):
    return VerificationIntent(collection='accounts', company='Acme Corp',
                              requested_fields=fields, require_new_record=False)


def account_result():
    return {key: next(row for row in snapshot_state()['accounts'] if row['customer_name']=='Acme Corp')[key]
            for key in ('customer_name', 'tier', 'mrr', 'account_manager', 'status')}


async def test_persisted_invoice_reaches_intent_directly_and_interprets_once(workspace):
    provider = ContractProvider(VerificationIntent(collection='invoices', company='Acme Corp', selection='latest'))
    verifier = CapabilityVerifier(provider)
    before = snapshot_state()
    persist_invoice()
    task = Subtask(task_id='invoice', goal='Record latest invoice', success_criteria=['Source matches Finance'],
                   verification_capability='browser')
    for _ in range(2):
        result = await verifier.verify('Record the latest Acme Corp invoice', task, before, ContextMemory())
        assert result.verified, result
        assert result.evidence['state_delta'] == state_delta(before, snapshot_state())
    assert len(provider.schemas) == 1 and issubclass(provider.schemas[0], VerificationIntent)
    assert not provider.responses


@pytest.mark.parametrize('capability', ['documents', 'python'])
async def test_read_only_capability_rejects_business_write_before_model_call(workspace, capability):
    provider = FakeProvider([])
    verifier = CapabilityVerifier(provider)
    before = snapshot_state()
    persist_invoice()
    task = Subtask(task_id='write', goal='Read source', success_criteria=['Sourced'],
                   verification_capability=capability, result={'answer':'done'})
    result = await verifier.verify('Read source', task, before, ContextMemory())
    assert result.outcome == 'FATAL_FAILURE' and not result.verified
    assert result.evidence['state_delta']['invoices']['created']
    assert not provider.call_history


async def test_valid_quote_is_insufficient_for_an_unsupported_answer(workspace):
    chunk = chunks('complaint_4821.txt')[0]
    provider = FakeProvider([{'covered':False, 'reason':'Wrong customer', 'material_claims':['Customer is Other'],
                             'unsupported_claims':['Wrong customer']}])
    task = Subtask(task_id='answer', goal='Identify customer', success_criteria=['Customer sourced'],
        verification_capability='documents', result={'customer':'Other'},
        evidence=[SourceReference(document_id=chunk['document_id'], chunk_id=chunk['chunk_id'], quote=chunk['text'][:50])])
    result = await CapabilityVerifier(provider).verify('Identify customer', task, snapshot_state(), ContextMemory())
    assert not result.verified
    assert len(provider.call_history) == 1


async def test_account_lookup_is_read_only_and_exact(workspace):
    provider = ContractProvider(account_contract(['customer_name', 'tier', 'mrr', 'account_manager', 'status']))
    result = await CapabilityVerifier(provider).verify('Read the Acme Corp account fields', account_task(account_result()),
                                                       snapshot_state(), ContextMemory())
    assert result.verified
    assert len(provider.schemas) == 1 and issubclass(provider.schemas[0], VerificationIntent)
    checks = result.evidence['verification_result']['criteria_results']
    assert len([check for check in checks if check['criterion'].startswith('CRM ')]) == 5


@pytest.mark.parametrize('field', ['customer_name', 'tier', 'mrr', 'account_manager', 'status'])
@pytest.mark.parametrize('missing', [True, False])
async def test_missing_or_wrong_crm_fields_fail(workspace, field, missing):
    answer = account_result()
    if missing:
        answer.pop(field)
    else:
        answer[field] = -1 if field == 'mrr' else 'Incorrect'
    provider = ContractProvider(account_contract(list(account_result())))
    result = await CapabilityVerifier(provider).verify('Read account fields', account_task(answer), snapshot_state(), ContextMemory())
    assert not result.verified
    assert any(field in discrepancy for discrepancy in result.discrepancies)


@pytest.mark.parametrize('mutation', ['create', 'update', 'delete'])
async def test_lookup_requires_zero_mutations(workspace, mutation):
    provider = ContractProvider(account_contract(['tier']))
    verifier = CapabilityVerifier(provider)
    before = snapshot_state()
    answer = account_result()
    if mutation == 'create':
        persist_invoice()
    else:
        with get_db_connection() as connection:
            statement = "UPDATE crm_accounts SET status='Inactive' WHERE customer_name='Globex Inc'" if mutation == 'update' else 'DELETE FROM support_tickets'
            connection.execute(statement)
            connection.commit()
    result = await verifier.verify('Read Acme Corp tier', account_task(answer), before, ContextMemory())
    assert not result.verified
    assert any('Unexpected records' in discrepancy for discrepancy in result.discrepancies)


@pytest.mark.parametrize('identity', [' acme   corp ', 'Missing customer'])
async def test_lookup_canonical_resolution_rejects_ambiguous_or_missing_accounts(workspace, identity):
    if identity.strip().lower().startswith('acme'):
        with get_db_connection() as connection:
            connection.execute("INSERT INTO crm_accounts(customer_name,tier,mrr,account_manager,status,created_at) SELECT 'ACME CORP',tier,mrr,account_manager,status,created_at FROM crm_accounts WHERE customer_name='Acme Corp'")
            connection.commit()
    intent = account_contract(['tier'])
    intent.company = identity
    result = await CapabilityVerifier(ContractProvider(intent)).verify('Read account tier', account_task(account_result()), snapshot_state(), ContextMemory())
    assert not result.verified
    assert any('ambiguous' in discrepancy for discrepancy in result.discrepancies)


async def test_canonical_account_identity_resolves_without_substring_matching(workspace):
    intent = account_contract(['tier'])
    intent.company = ' acme   CORP '
    result = await CapabilityVerifier(ContractProvider(intent)).verify('Read account tier', account_task(account_result()), snapshot_state(), ContextMemory())
    assert result.verified


async def test_conditional_ticket_noop_reaches_application_contract(workspace):
    provider = ContractProvider(VerificationIntent(collection='tickets', complaint_id='4822',
                                                   condition_tier='Enterprise', priority='High'))
    verifier = CapabilityVerifier(provider)
    before = snapshot_state()
    task = Subtask(task_id='conditional', goal='Create a ticket only for the requested tier',
                   success_criteria=['Condition respected'], verification_capability='browser')
    result = await verifier.verify('Create a ticket for complaint 4822 only if Enterprise', task, before, ContextMemory())
    assert result.verified
    checks = result.evidence['verification_result']['criteria_results']
    assert any(check['evidence'].get('conditional_outcome') == 'no_op' for check in checks)
    persist_invoice()
    assert not (await verifier.verify('Create a ticket for complaint 4822 only if Enterprise', task, before, ContextMemory())).verified
    assert len(provider.schemas) == 1 and issubclass(provider.schemas[0], VerificationIntent)


async def test_final_audit_rejects_state_changes_after_verification(workspace):
    before = snapshot_state()
    provider = ContractProvider(VerificationIntent(collection='invoices', company='Acme Corp', selection='latest'))
    verifier = CapabilityVerifier(provider)
    persist_invoice()
    task = Subtask(task_id='invoice', goal='Record latest invoice', success_criteria=['Recorded'], verification_capability='browser')
    result = await verifier.verify('Record latest invoice', task, before, ContextMemory())
    assert result.verified
    task.result['verification'] = result.model_dump()
    assert verifier.audit_mutations([task], before, snapshot_state()).verified
    with get_db_connection() as connection:
        connection.execute("UPDATE finance_invoices SET amount_minor=1 WHERE invoice_number='INV-1044'")
        connection.commit()
    assert not verifier.audit_mutations([task], before, snapshot_state()).verified
    assert len(provider.schemas) == 1 and issubclass(provider.schemas[0], VerificationIntent)


@pytest.mark.parametrize('defect', ['currency', 'amount', 'due_date', 'source', 'duplicate', 'unwanted_mutation'])
async def test_browser_path_preserves_exact_invoice_rejections(workspace, defect):
    provider = ContractProvider(VerificationIntent(collection='invoices', company='Acme Corp', selection='latest'))
    verifier = CapabilityVerifier(provider)
    before = snapshot_state()
    persist_invoice()
    statements = {
        'currency': "UPDATE finance_invoices SET currency='USD' WHERE invoice_number='INV-1044'",
        'amount': "UPDATE finance_invoices SET amount_minor=1 WHERE invoice_number='INV-1044'",
        'due_date': "UPDATE finance_invoices SET due_date='2026-11-01' WHERE invoice_number='INV-1044'",
        'source': "UPDATE finance_invoices SET source_reference='unrelated.pdf' WHERE invoice_number='INV-1044'",
        'duplicate': "INSERT INTO finance_invoices(company,invoice_number,amount_minor,currency,due_date,status,created_at,source_reference) SELECT 'ACME CORP',invoice_number,amount_minor,currency,due_date,status,created_at,source_reference FROM finance_invoices WHERE invoice_number='INV-1044'",
        'unwanted_mutation': "UPDATE crm_accounts SET status='Inactive' WHERE customer_name='Globex Inc'",
    }
    with get_db_connection() as connection:
        if defect == 'duplicate':
            # Simulate an imported corrupt database despite the normal uniqueness guard.
            connection.execute('DROP INDEX finance_canonical_identity')
        connection.execute(statements[defect])
        connection.commit()
    task = Subtask(task_id='invoice', goal='Record latest invoice', success_criteria=['Source matches Finance'],
                   verification_capability='browser')
    result = await verifier.verify('Record latest invoice', task, before, ContextMemory())
    assert not result.verified
    assert len(provider.schemas) == 1 and issubclass(provider.schemas[0], VerificationIntent)
