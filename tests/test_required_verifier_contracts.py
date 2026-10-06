"""Actual provider payloads and held-out request literals, without network calls."""
from copy import deepcopy
import json

import httpx
import pytest
from jsonschema import Draft202012Validator

from backend.app.agent.provider import GeminiProvider, FakeProvider
from backend.app.agent.schemas import VerificationIntent
from backend.app.agent.verifier import VerifierEngine
from backend.app.capabilities.python.verification import CalculationContract, calculation_schema
from test_verification_interpretation import workspace, COMPLAINT, DATASET, GOOD_TICKET, BAD_TICKET, CALCULATION


def assert_invalid(schema, contract):
    assert list(Draft202012Validator(schema).iter_errors(contract))


def test_ticket_wire_requires_all_fields_and_explicit_prerequisites(workspace):
    schema = VerifierEngine(workspace, FakeProvider([])).generation_schema(COMPLAINT).model_json_schema()
    Draft202012Validator.check_schema(schema)
    assert set(schema['required']) == set(schema['properties'])
    assert all('default' not in field for field in schema['properties'].values())
    Draft202012Validator(schema).validate(GOOD_TICKET.model_dump())
    for field in ['complaint_id', 'priority', 'condition_tier']:
        bad = GOOD_TICKET.model_dump();bad[field] = None
        assert_invalid(schema, bad)
        bad = GOOD_TICKET.model_dump();bad.pop(field)
        assert_invalid(schema, bad)
    for field, value in [('complaint_id','4822'), ('priority','Low'), ('condition_tier','Growth'), ('require_new_record',False)]:
        bad = GOOD_TICKET.model_dump();bad[field] = value
        assert_invalid(schema, bad)


def test_ticket_wire_is_not_specific_to_the_demo_customer_or_tier(workspace):
    objective = 'Read complaint 4822 and create a medium-priority support ticket if the customer is a Starter customer.'
    schema = VerifierEngine(workspace, FakeProvider([])).generation_schema(objective).model_json_schema()
    intent = VerificationIntent(collection='tickets', complaint_id='4822', priority='Medium', condition_tier='Starter')
    Draft202012Validator(schema).validate(intent.model_dump())
    assert_invalid(schema, GOOD_TICKET.model_dump())
    assert 'Acme' not in json.dumps(schema) and 'Enterprise' not in json.dumps(schema)


def test_latest_invoice_cannot_be_weakened_to_specific_or_existing(workspace):
    objective = 'Find the latest invoice from Example Ltd and enter it into Finance.'
    schema = VerifierEngine(workspace, FakeProvider([])).generation_schema(objective).model_json_schema()
    valid = VerificationIntent(collection='invoices', company='Example Ltd', selection='latest').model_dump()
    Draft202012Validator(schema).validate(valid)
    assert_invalid(schema, {**valid, 'selection':'specific', 'invoice_number':'A-1'})
    assert_invalid(schema, {**valid, 'require_new_record':False})


def test_negated_creation_and_latest_selection_do_not_become_positive_requirements(workspace):
    objective = "Do not create an invoice. Check invoice A-1, not the latest invoice, for Example Ltd."
    schema = VerifierEngine(workspace, FakeProvider([])).generation_schema(objective).model_json_schema()
    valid = VerificationIntent(collection='invoices', company='Example Ltd', selection='specific',
                               invoice_number='A-1', require_new_record=False)
    Draft202012Validator(schema).validate(valid.model_dump())


@pytest.mark.parametrize('objective,contract', [
    ('Read the tier for Example Ltd', VerificationIntent(collection='accounts', company='Example Ltd', requested_fields=['tier'], require_new_record=False)),
    ('Read complaint 91 and create a support ticket', VerificationIntent(collection='tickets', complaint_id='91')),
])
def test_read_only_and_unconditional_contracts_remain_representable(workspace, objective, contract):
    schema = VerifierEngine(workspace, FakeProvider([])).generation_schema(objective).model_json_schema()
    Draft202012Validator(schema).validate(contract.model_dump())


def test_decline_wire_requires_ordered_periods_and_explicit_direction():
    schema = calculation_schema(DATASET, {'sales.csv'}, {'region','decline'}).model_json_schema()
    Draft202012Validator.check_schema(schema)
    assert set(schema['required']) == set(schema['properties'])
    assert all('default' not in field for field in schema['properties'].values())
    good = CalculationContract(**CALCULATION).model_dump()
    Draft202012Validator(schema).validate(good)
    for changes in [{'baseline_period':None}, {'current_period':None}, {'period_column':None},
                    {'convention':'current_minus_baseline'}, {'selection':'min'}, {'measure':'percent_change'},
                    {'baseline_period':'2026-09','current_period':'2026-08'}, {'document_id':'other.csv'}]:
        assert_invalid(schema, {**good,**changes})


def test_other_calculations_and_unspecified_units_are_not_forced_to_decline():
    objective = 'Find the market with the largest percent growth from 2024-01 to 2024-02 in input.csv'
    schema = calculation_schema(objective, {'input.csv'}, {'market','growth'}).model_json_schema()
    good = CalculationContract(document_id='input.csv', group_column='market', value_column='amount',
        measure='percent_change', convention='current_minus_baseline', period_column='period',
        baseline_period='2024-01', current_period='2024-02', selection='max', group_metric='market', value_metric='growth')
    Draft202012Validator(schema).validate(good.model_dump())
    aggregate = good.model_copy(update={'measure':'value','period_column':None,'baseline_period':None,'current_period':None})
    aggregation_schema = calculation_schema('Find the market with largest total amount', {'input.csv'}, {'market','growth'}).model_json_schema()
    Draft202012Validator(aggregation_schema).validate(aggregate.model_dump())


@pytest.mark.anyio
async def test_actual_gemini_payload_uses_required_schema_and_repair_never_dispatches(workspace, tmp_path):
    requests = []
    def send(request):
        requests.append(json.loads(request.content))
        value = BAD_TICKET.model_dump() if len(requests)==1 else GOOD_TICKET.model_dump()
        return httpx.Response(200,json={'status':'completed','steps':[{'type':'model_output','content':[{'type':'text','text':json.dumps(value)}]}],
            'usage':{'total_input_tokens':1,'total_output_tokens':1,'total_tokens':2}})
    provider = GeminiProvider(api_key='test-only', transport=httpx.MockTransport(send), rate_db_path=tmp_path/'limits.db')
    verifier = VerifierEngine(workspace, provider)
    intent = await verifier.interpret(COMPLAINT, [COMPLAINT])
    assert intent == GOOD_TICKET and len(requests)==2
    wire = requests[0]['response_format']['schema']
    assert wire == requests[1]['response_format']['schema']
    assert set(wire['required']) == set(wire['properties'])
    assert_invalid(wire, BAD_TICKET.model_dump())
    Draft202012Validator(wire).validate(intent.model_dump())
    assert COMPLAINT in requests[0]['input'] and 'interpretation_repair' in requests[1]['input']
    assert [attempt['admitted'] for attempt in verifier.interpretation_attempts] == [False,True]
    # An admitted interpretation is reused; no fresh generation or cache reset.
    assert await verifier.interpret(COMPLAINT,[COMPLAINT]) == intent and len(requests)==2
