"""Gemini wire completeness, selected-tool repairs and shared-contract invariants."""
import json
import httpx
import pytest
from backend.app.agent.provider import GeminiProvider, ProviderError
from backend.app.agent.schemas import AgentDecision
from backend.app.agent_v2.state import Decision
from backend.app.capabilities.registry import build_registry

pytestmark = pytest.mark.anyio


def decision(action='act', tool='inspect_file', args=None, **values):
    return {'thought': 'Use the current observation.', 'action': action,
            'tool_name': tool, 'tool_args': args if args is not None else {'document_id': 'example.pdf'},
            'clarification_question': None, 'failure_reason': None,
            'evidence': [], 'result': {}, 'replan': False, **values}


def provider(tmp_path, outputs):
    requests = []
    def send(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={'status': 'completed', 'steps': [
            {'type': 'model_output', 'content': [{'type': 'text', 'text': json.dumps(outputs[len(requests)-1])}]}],
            'usage': {'total_input_tokens': 3, 'total_output_tokens': 4, 'total_tokens': 7}})
    adapter = GeminiProvider(api_key='test-secret', transport=httpx.MockTransport(send), rate_db_path=tmp_path/'limits.db')
    adapter.configure_tools(build_registry().get_schemas())
    return adapter, requests


@pytest.mark.parametrize('schema', [AgentDecision, Decision])
async def test_complete_wire_shape_and_valid_act_need_one_request(tmp_path, schema):
    adapter, requests = provider(tmp_path, [decision()])
    wire = adapter._wire_schema(schema)
    assert set(wire['required']) == set(decision())
    assert wire['properties']['tool_args']['type'] == 'object'
    assert 'anyOf' not in wire['properties']['tool_args']
    result, metadata = await adapter.generate_structured([], schema)
    assert result.tool_args == {'document_id': 'example.pdf'}
    assert len(requests) == metadata['attempts'] == 1
    assert metadata['schema_repairs'] == 0


async def test_zero_argument_tool_cannot_validate_inspect_file_arguments(tmp_path):
    adapter, _ = provider(tmp_path, [])
    adapter._validate(decision(tool='browser_back', args={}), Decision)
    with pytest.raises(ValueError, match='document_id.*required'):
        adapter._validate(decision(args={}), Decision)


@pytest.mark.parametrize(('tool', 'args'), [
    ('inspect_file', {'document_id': 'example.pdf'}),
    ('profile_dataset', {'document_id': 'example.csv'}),
    ('browser_open', {'url': '/workspace/crm'}),
])
@pytest.mark.parametrize('invalid', ['missing', 'empty', 'null', 'wrong_type', 'extra'])
async def test_invalid_arguments_narrow_existing_repair_to_exact_selected_tool(tmp_path, tool, args, invalid):
    first = decision(tool=tool, args={})
    if invalid == 'missing':
        first.pop('tool_args')
    elif invalid == 'null':
        first['tool_args'] = None
    elif invalid == 'wrong_type':
        first['tool_args'] = {key: 7 for key in args}
    elif invalid == 'extra':
        first['tool_args'] = {**args, 'invented': True}
    adapter, requests = provider(tmp_path, [first, decision(tool=tool, args=args)])
    result, metadata = await adapter.generate_structured([], Decision)
    expected = next(t['parameters'] for t in adapter.tool_schemas if t['name'] == tool)
    repair_schema = requests[1]['response_format']['schema']
    assert repair_schema['properties']['tool_name'] == {'type': 'string', 'enum': [tool]}
    assert repair_schema['properties']['action'] == {'type': 'string', 'enum': ['act']}
    assert repair_schema['properties']['tool_args'] == expected
    assert {'tool_name', 'tool_args'} <= set(repair_schema['required'])
    repair = json.loads(requests[1]['input'])
    assert json.loads(repair['previous_output']) == first
    assert repair['repair']['selected_tool'] == tool
    assert repair['repair']['tool_args_schema'] == expected
    assert repair['repair']['validation_error']
    assert 'COMPLETE' in repair['repair']['instruction']
    assert 'do not change the selected tool' in repair['repair']['instruction']
    assert result.tool_name == tool and result.tool_args == args
    assert len(requests) == metadata['attempts'] == 2
    assert metadata['schema_repairs'] == 1
    assert adapter.usage_snapshot()['requests'] == 2
    assert adapter.usage_snapshot()['total_tokens'] == 14


async def test_repair_cannot_switch_to_another_registered_tool(tmp_path):
    adapter, requests = provider(tmp_path, [decision(args={}), decision(tool='browser_back', args={})])
    with pytest.raises(ProviderError, match='after one repair'):
        await adapter.generate_structured([], Decision)
    assert len(requests) == 2


@pytest.mark.parametrize('schema', [AgentDecision, Decision])
@pytest.mark.parametrize(('action', 'values'), [
    ('ready_for_verification', {}),
    ('need_clarification', {'clarification_question': 'Which document?'}),
    ('fail', {'failure_reason': 'The source is unavailable.'}),
])
async def test_non_act_stable_wire_shape_preserves_shared_invariants_without_repair(tmp_path, schema, action, values):
    raw = decision(action=action, tool=None, args={}, **values)
    adapter, requests = provider(tmp_path, [raw])
    result, metadata = await adapter.generate_structured([], schema)
    assert len(requests) == 1 and metadata['schema_repairs'] == 0
    assert raw['tool_name'] is None and raw['tool_args'] == {}
    assert result.tool_name is None and result.tool_args is None
    assert schema.model_validate(result.model_dump()) == result


@pytest.mark.parametrize('raw', [
    decision(action='ready_for_verification', tool='browser_back', args={}),
    decision(action='ready_for_verification', tool=None, args={'url': '/workspace/crm'}),
    decision(action='need_clarification', tool=None, args={}),
    decision(action='fail', tool=None, args={}),
])
async def test_non_act_invalid_tools_or_missing_reason_still_fail(tmp_path, raw):
    adapter, _ = provider(tmp_path, [])
    with pytest.raises(ValueError):
        adapter._validate(raw, Decision)


async def test_shared_and_groq_schemas_keep_their_original_required_fields():
    assert AgentDecision.model_json_schema()['required'] == ['thought', 'action']
    assert Decision.model_json_schema()['required'] == ['thought', 'action']
    with pytest.raises(ValueError, match='non-act decisions cannot include tools'):
        AgentDecision.model_validate({'thought': 'Done', 'action': 'ready_for_verification', 'tool_args': {}})
