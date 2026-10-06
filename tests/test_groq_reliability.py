"""Groq scheduling and structured-repair regressions; all transport is mocked."""
import asyncio
import json
from pathlib import Path
import time

import httpx
import pytest

from backend.app.agent.provider import GroqProvider, ProviderError
from backend.app.agent_v2.state import Decision


VALID = {'thought': 'Use the observed form.', 'action': 'act',
         'tool_name': 'browser_type',
         'tool_args': {'element_id': '@ticket_summary', 'text': 'Grounded summary'}}
DUPLICATE = ('{"thought":"Fill the form","action":"act",'
             '"tool_name":"browser_type","tool_args":{"element_id":"@ticket_summary","text":"First"},'
             '"tool_name":null,"tool_args":null}')


def response(content, cached=None, headers=None):
    usage = {'prompt_tokens': 1200, 'completion_tokens': 200}
    if cached is not None:
        usage['prompt_tokens_details'] = {'cached_tokens': cached}
    return httpx.Response(200, headers=headers, json={
        'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}], 'usage': usage})


def tracked_transport(outputs, calls):
    def send(request):
        calls.append(json.loads(request.content))
        assert len(calls) <= len(outputs), 'Unexpected additional HTTP request'
        return outputs[len(calls) - 1]
    return httpx.MockTransport(send)


def paced_client(monkeypatch, transport):
    original = httpx.AsyncClient
    waits = []
    async def sleep(delay):
        waits.append(delay)
    monkeypatch.setattr(asyncio, 'sleep', sleep)
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kwargs:
                        original(timeout=kwargs['timeout'], transport=transport))
    monkeypatch.setattr(GroqProvider, '_rate_state', {})
    return waits


@pytest.mark.anyio
async def test_exact_historical_8100_estimate_reaches_transport(monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'tests/fixtures/regressions/provider_repair/provider-block-evidence.json'
    historical = json.loads(path.read_text())['repair_payload_reconstructed_offline']
    assert len(json.dumps(historical, separators=(',', ':'))) / 3.5 + 2048 == pytest.approx(8100.571428571428)
    calls = []
    waits = paced_client(monkeypatch, tracked_transport([response(json.dumps(VALID))], calls))
    provider = GroqProvider(api_key='test-secret', max_output_tokens=2048)
    GroqProvider._rate_state[(provider.base_url, provider.model)] = (8000, 8000, time.monotonic(), 0)
    decision, meta = await provider.generate_structured(historical['messages'], Decision)
    assert calls == [historical]
    assert decision.tool_name == 'browser_type' and not waits
    assert meta['scheduling_estimates'][0]['estimated_tokens'] == pytest.approx(8100.571428571428)
    assert meta['scheduling_estimates'][0]['exceeds_limit'] is True


@pytest.mark.anyio
async def test_historical_duplicate_response_can_repair_with_original_context(monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'tests/fixtures/regressions/provider_repair/complaint-http.json'
    historical = json.loads(path.read_text())[-1]
    original_content = historical['response']['choices'][0]['message']['content']
    calls = []
    waits = paced_client(monkeypatch, tracked_transport([
        response(original_content), response(json.dumps(VALID))], calls))
    provider = GroqProvider(api_key='test-secret')
    GroqProvider._rate_state[(provider.base_url, provider.model)] = (8000, 8000, time.monotonic(), 0)
    decision, meta = await provider.generate_structured(historical['messages'], Decision)
    assert len(calls) == 2 and not waits
    assert calls[1]['messages'][:-2] == historical['messages']
    assert calls[1]['messages'][-2]['content'] == original_content
    assert decision.tool_args['text'] == 'Grounded summary'
    assert meta['response_errors'] == [{'code': 'DUPLICATE_JSON_KEYS', 'keys': ['tool_args', 'tool_name']}]
    assert meta['request_completion_caps'] == [2048, 1024]


@pytest.mark.anyio
async def test_oversized_estimate_defers_to_actual_429(monkeypatch):
    calls = []
    waits = paced_client(monkeypatch, tracked_transport([
        httpx.Response(429, headers={'retry-after': '0'}),
        httpx.Response(429, headers={'retry-after': '0'})], calls))
    provider = GroqProvider(api_key='test-secret')
    GroqProvider._rate_state[(provider.base_url, provider.model)] = (8000, 8000, time.monotonic(), 0)
    with pytest.raises(ProviderError, match='Groq HTTP 429; retry budget exhausted'):
        await provider.generate_structured([{'role': 'user', 'content': 'X' * 30000}], Decision)
    assert len(calls) == 2 and waits == [0]


@pytest.mark.anyio
async def test_advisory_estimate_waits_once_for_observed_reset(monkeypatch):
    calls = []
    waits = paced_client(monkeypatch, tracked_transport([response(json.dumps(VALID))], calls))
    provider = GroqProvider(api_key='test-secret')
    GroqProvider._rate_state[(provider.base_url, provider.model)] = (8000, 0, time.monotonic(), 2.75)
    _, meta = await provider.generate_structured([{'role': 'user', 'content': 'Inspect'}], Decision)
    assert len(calls) == len(waits) == 1
    assert waits[0] == pytest.approx(2.75, abs=.1)
    assert meta['rate_wait_seconds'] == pytest.approx(waits[0], abs=.001)


@pytest.mark.anyio
async def test_oversized_estimate_waits_bounded_then_sends(monkeypatch):
    calls = []
    waits = paced_client(monkeypatch, tracked_transport([response(json.dumps(VALID))], calls))
    provider = GroqProvider(api_key='test-secret')
    GroqProvider._rate_state[(provider.base_url, provider.model)] = (8000, 0, time.monotonic(), 120)
    _, meta = await provider.generate_structured([{'role': 'user', 'content': 'X' * 30000}], Decision)
    assert len(calls) == 1 and waits == [60]
    assert meta['scheduling_estimates'][0]['exceeds_limit']


@pytest.mark.anyio
async def test_header_reset_tracking(monkeypatch):
    calls = []
    headers = {'x-ratelimit-limit-tokens': '8000', 'x-ratelimit-remaining-tokens': '3000',
               'x-ratelimit-reset-tokens': '1m2.5s'}
    paced_client(monkeypatch, tracked_transport([response(json.dumps(VALID), headers=headers)], calls))
    provider = GroqProvider(api_key='test-secret')
    await provider.generate_structured([], Decision)
    limit, remaining, _, reset = GroqProvider._rate_state[(provider.base_url, provider.model)]
    assert (limit, remaining, reset) == (8000, 3000, 62.5)


@pytest.mark.anyio
@pytest.mark.parametrize('headers,message,delay', [
    ({'retry-after': '1.5', 'x-ratelimit-reset-tokens': '20s'}, '', 1.5),
    ({'x-ratelimit-reset-tokens': '2.5s'}, 'tokens per minute (TPM)', 2.5),
    ({'x-ratelimit-reset-requests': '3s', 'x-ratelimit-reset-tokens': '20s'}, 'requests per minute (RPM)', 3),
    ({'retry-after': '100'}, '', 60),
    ({'retry-after': 'invalid', 'x-ratelimit-reset-tokens': '4s'}, '', 4),
    ({}, '', .25),
    ({'x-ratelimit-reset-tokens': '2s'}, {'malformed': 'message'}, 2),
])
async def test_real_429_headers_and_single_retry(monkeypatch, headers, message, delay):
    waits, calls = [], []
    async def sleep(value):
        waits.append(value)
    monkeypatch.setattr(asyncio, 'sleep', sleep)
    outputs = [httpx.Response(429, headers=headers, json={'error': {'message': message}}),
               response(json.dumps(VALID))]
    provider = GroqProvider(api_key='test-secret', transport=tracked_transport(outputs, calls))
    _, meta = await provider.generate_structured([], Decision)
    assert len(calls) == 2 and waits == [delay]
    assert meta['transport_retries'] == 1
    assert meta['rate_wait_seconds'] == delay


@pytest.mark.anyio
async def test_persistent_429_stops_after_one_retry(monkeypatch):
    waits, calls = [], []
    async def sleep(delay):
        waits.append(delay)
    monkeypatch.setattr(asyncio, 'sleep', sleep)
    outputs = [httpx.Response(429, headers={'x-ratelimit-reset-tokens': '2s'}) for _ in range(2)]
    provider = GroqProvider(api_key='test-secret', transport=tracked_transport(outputs, calls))
    with pytest.raises(ProviderError, match='429; retry budget exhausted'):
        await provider.generate_structured([], Decision)
    assert len(calls) == 2 and waits == [2]


@pytest.mark.anyio
async def test_duplicates_repaired_once_without_selecting_values():
    calls = []
    provider = GroqProvider(api_key='test-secret', transport=tracked_transport([
        response(DUPLICATE), response(json.dumps(VALID))], calls))
    decision, meta = await provider.generate_structured([], Decision)
    assert len(calls) == 2 and decision.tool_args['text'] == 'Grounded summary'
    assert meta['schema_repairs'] == 1
    assert meta['response_errors'] == [{'code': 'DUPLICATE_JSON_KEYS', 'keys': ['tool_args', 'tool_name']}]
    assert calls[0]['max_completion_tokens'] == 2048
    assert calls[1]['max_completion_tokens'] == 1024
    assert meta['repair_max_completion_tokens'] == 1024
    assert meta['request_completion_caps'] == [2048, 1024]
    assert all(call['response_format']['json_schema']['strict'] is False for call in calls)
    assert all(call['response_format']['json_schema']['schema'] == Decision.model_json_schema() for call in calls)
    correction = calls[1]['messages'][-1]['content']
    assert 'DUPLICATE_JSON_KEYS' in correction and 'First' not in correction
    assert calls[1]['messages'][-2]['content'] == DUPLICATE


@pytest.mark.anyio
async def test_duplicates_rejected_even_when_last_values_are_valid():
    duplicate = ('{"thought":"Use form","action":"act","tool_name":null,"tool_args":null,'
                 '"tool_name":"browser_type","tool_args":{"element_id":"@ticket_summary","text":"Last"}}')
    calls = []
    provider = GroqProvider(api_key='test-secret', transport=tracked_transport([
        response(duplicate), response(json.dumps(VALID))], calls))
    decision, _ = await provider.generate_structured([], Decision)
    assert len(calls) == 2 and decision.tool_args['text'] != 'Last'


@pytest.mark.anyio
@pytest.mark.parametrize('bad', [DUPLICATE, '{"thought":"Still invalid","action":"act"}', 'not json'])
async def test_failed_repair_is_terminal_without_third_request(bad):
    calls = []
    provider = GroqProvider(api_key='test-secret', transport=tracked_transport([
        response(DUPLICATE), response(bad)], calls))
    with pytest.raises(ProviderError, match='after one repair') as error:
        await provider.generate_structured([], Decision)
    assert len(calls) == 2
    assert 'test-secret' not in str(error.value) and '"First"' not in str(error.value)


def test_nested_duplicate_keys_and_separate_objects():
    assert GroqProvider._duplicate_json_keys('{"items":[{"a":1,"a":2},{"b":{"c":0,"c":1}}]}') == ['a', 'c']
    assert GroqProvider._duplicate_json_keys('{"items":[{"a":1},{"a":2}]}') == []


@pytest.mark.anyio
@pytest.mark.parametrize('cached', [None, 0, 1000])
async def test_only_observed_cache_metadata_is_preserved(cached):
    calls = []
    provider = GroqProvider(api_key='test-secret', transport=tracked_transport([
        response(json.dumps(VALID), cached=cached)], calls))
    _, meta = await provider.generate_structured([], Decision)
    if cached is None:
        assert 'cached_tokens' not in meta and 'cached_tokens' not in provider.usage_snapshot()
    else:
        assert meta['cached_tokens'] == provider.usage_snapshot()['cached_tokens'] == cached
    assert meta['prompt_eval_count'] == 1200


@pytest.mark.anyio
async def test_cache_counts_include_invalid_and_repaired_responses():
    calls = []
    provider = GroqProvider(api_key='test-secret', transport=tracked_transport([
        response(DUPLICATE, cached=100), response(json.dumps(VALID), cached=500)], calls))
    _, meta = await provider.generate_structured([], Decision)
    assert meta['cached_tokens'] == provider.usage_snapshot()['cached_tokens'] == 600
    assert meta['prompt_eval_count'] == 2400 and meta['eval_count'] == 400


@pytest.mark.anyio
async def test_malformed_cache_details_do_not_invent_counts():
    calls = []
    output = response(json.dumps(VALID))
    body = output.json()
    body['usage']['prompt_tokens_details'] = 'invalid metadata'
    provider = GroqProvider(api_key='test-secret', transport=tracked_transport([
        httpx.Response(200, json=body)], calls))
    _, meta = await provider.generate_structured([], Decision)
    assert 'cached_tokens' not in meta and 'cached_tokens' not in provider.usage_snapshot()


@pytest.mark.anyio
async def test_upstream_contract_repair_has_same_smaller_cap():
    calls = []
    provider = GroqProvider(api_key='test-secret', transport=tracked_transport([
        httpx.Response(400, json={'error': {'code': 'tool_use_failed', 'failed_generation': '{}'}}),
        response(json.dumps(VALID))], calls))
    _, meta = await provider.generate_structured([], Decision)
    assert meta['schema_repairs'] == 1 and meta['request_completion_caps'] == [2048, 1024]


@pytest.mark.anyio
@pytest.mark.parametrize('limit', ['tokens per day (TPD)', 'requests per day (RPD)'])
async def test_daily_limits_never_retry(limit):
    calls = []
    provider = GroqProvider(api_key='test-secret', transport=tracked_transport([
        httpx.Response(429, headers={'retry-after': '1200'}, json={'error': {'message': limit}})], calls))
    with pytest.raises(ProviderError, match='daily quota exceeded'):
        await provider.generate_structured([], Decision)
    assert len(calls) == 1
