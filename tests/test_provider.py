"""Groq protocol, retry bounds and secret-free error contracts without API calls."""
import json
import httpx
import pytest
from backend.app.agent.provider import GroqProvider, ProviderError
from backend.app.agent.schemas import TaskPlan

PLAN={'objective':'Check','success_criteria':['Verified'],'strategy':['Inspect']}

def response(content, **kwargs):
    return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps(content)},'finish_reason':'stop'}],'usage':{'prompt_tokens':3,'completion_tokens':4}},**kwargs)

@pytest.mark.anyio
async def test_groq_schema_and_rate_limit_retry_bound():
    calls=[]
    def send(request):
        calls.append(json.loads(request.content))
        return httpx.Response(429,headers={'retry-after':'0'}) if len(calls)==1 else response(PLAN)
    provider=GroqProvider(api_key='test-secret',transport=httpx.MockTransport(send))
    plan,meta=await provider.generate_structured([{'role':'user','content':'Check'}],TaskPlan)
    assert plan.objective=='Check'
    assert meta['provider']=='groq' and meta['attempts']==2 and meta['transport_retries']==1
    assert calls[0]['response_format']['type']=='json_schema'
    assert calls[0]['tool_choice']=='none'
    assert calls[0]['max_completion_tokens']==2048 and calls[0]['reasoning_effort']=='medium'
    assert 'test-secret' not in json.dumps(meta)

@pytest.mark.anyio
async def test_persistent_rate_limit_and_timeout_never_expose_secret():
    calls=[]
    def send(request):
        calls.append(request)
        return httpx.Response(429,text='test-secret',headers={'retry-after':'0'})
    provider=GroqProvider(api_key='test-secret',transport=httpx.MockTransport(send))
    with pytest.raises(ProviderError) as error:
        await provider.generate_structured([],TaskPlan)
    assert len(calls)==2 and 'test-secret' not in str(error.value)
    def timeout(request):
        raise httpx.ReadTimeout('test-secret',request=request)
    with pytest.raises(ProviderError,match='timed out') as error:
        await GroqProvider(api_key='test-secret',transport=httpx.MockTransport(timeout)).generate_structured([],TaskPlan)
    assert 'test-secret' not in str(error.value)


@pytest.mark.anyio
async def test_daily_quota_fails_immediately_without_leaking_upstream_message():
    calls = []
    def send(request):
        calls.append(request)
        return httpx.Response(429, json={'error': {'message':
            'Account test-secret on tokens per day (TPD): Limit 200000, Used 199893'}},
            headers={'retry-after': '1200'})
    with pytest.raises(ProviderError, match='daily quota exceeded') as error:
        await GroqProvider(api_key='test-secret', transport=httpx.MockTransport(send)).generate_structured([], TaskPlan)
    assert len(calls) == 1 and not error.value.retriable
    assert 'test-secret' not in str(error.value)

@pytest.mark.anyio
async def test_invalid_schema_repaired_once():
    calls=[]
    def send(request):
        calls.append(json.loads(request.content))
        return response({'unexpected':'value'})
    with pytest.raises(ProviderError,match='after one repair'):
        await GroqProvider(api_key='test',transport=httpx.MockTransport(send)).generate_structured([],TaskPlan)
    assert len(calls)==2
    assert 'Repair only the JSON' in calls[1]['messages'][-1]['content']

@pytest.mark.anyio
async def test_transport_retry_and_explicit_json_fallback():
    calls=[]
    def send(request):
        calls.append(json.loads(request.content))
        if len(calls)==1: raise httpx.ConnectError('failure',request=request)
        if len(calls)==2: return httpx.Response(400,text='json_schema not supported')
        return response(PLAN)
    _,meta=await GroqProvider(api_key='test',transport=httpx.MockTransport(send)).generate_structured([],TaskPlan)
    assert len(calls)==3 and meta['response_format']=='json_object'
    assert calls[-1]['response_format']=={'type':'json_object'}

@pytest.mark.anyio
async def test_provider_paces_requests_from_observed_headers(monkeypatch):
    import asyncio
    import time
    client_class=httpx.AsyncClient
    waits=[]
    async def sleep(delay): waits.append(delay)
    def send(request):
        return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps(PLAN)},'finish_reason':'stop'}]},headers={'x-ratelimit-limit-tokens':'8000','x-ratelimit-remaining-tokens':'3000'})
    monkeypatch.setattr(httpx,'AsyncClient',lambda **kwargs:client_class(timeout=kwargs['timeout'],transport=httpx.MockTransport(send)))
    monkeypatch.setattr(asyncio,'sleep',sleep)
    monkeypatch.setattr(GroqProvider,'_rate_state',{})
    provider=GroqProvider(api_key='test-secret')
    GroqProvider._rate_state[(provider.base_url,provider.model)]=(8000,0,time.monotonic())
    _,meta=await provider.generate_structured([{'role':'user','content':'Check'}],TaskPlan)
    assert len(waits)==1 and 0<waits[0]<=60 and meta['rate_wait_seconds']>0

@pytest.mark.anyio
async def test_upstream_schema_generation_failure_has_one_repair():
    calls=[]
    def send(request):
        calls.append(json.loads(request.content))
        return httpx.Response(400,json={'error':{'code':'json_validate_failed','failed_generation':'{}'}}) if len(calls)==1 else response(PLAN)
    _,meta=await GroqProvider(api_key='test-secret',transport=httpx.MockTransport(send)).generate_structured([],TaskPlan)
    assert len(calls)==2 and meta['schema_repairs']==1
    def always_invalid(request): return httpx.Response(400,json={'error':{'code':'json_validate_failed','failed_generation':'{}'}})
    with pytest.raises(ProviderError,match='json_validate_failed'):
        await GroqProvider(api_key='test-secret',transport=httpx.MockTransport(always_invalid)).generate_structured([],TaskPlan)

@pytest.mark.anyio
async def test_native_tool_generation_is_repaired_not_dispatched():
    calls=[]
    def send(request):
        calls.append(json.loads(request.content))
        if len(calls)==1:
            return httpx.Response(400,json={'error':{'code':'tool_use_failed','failed_generation':'<function=browser_open>{}</function>'}})
        return response(PLAN)
    plan,meta=await GroqProvider(api_key='test',transport=httpx.MockTransport(send)).generate_structured([],TaskPlan)
    assert plan.objective=='Check' and len(calls)==2 and meta['schema_repairs']==1
    assert 'Do not issue native API' in calls[-1]['messages'][-1]['content']
