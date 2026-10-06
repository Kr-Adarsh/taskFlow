"""
Replaceable LLM Provider interface with adapters for Groq and Gemini (Interactions API).
Keeps transport, request schemas, timeouts, and error normalization separate
from the business runtime loop.
"""

from abc import ABC, abstractmethod
import asyncio
import json
import hashlib
import math
import sqlite3
import os
from pathlib import Path
import re
import time
from typing import Any, Type, TypeVar, Optional
import httpx
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

def load_env_var(key: str, default: str = "") -> str:
    """Helper to get env var from os.environ or project .env file."""
    val = os.environ.get(key)
    if val:
        return val
    env_path = Path(__file__).resolve().parents[3] / ".env"
    if env_path.exists():
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        if k.strip() == key:
                            return v.strip().strip("'\"")
        except Exception:
            pass
    return default


def provider_model(provider, default):
    scoped = load_env_var(provider.upper() + '_MODEL')
    shared = load_env_var('LLM_MODEL')
    if scoped:
        return scoped
    if shared and shared.startswith('gemini-') == (provider == 'gemini'):
        return shared
    return default


class ProviderError(Exception):
    def __init__(self, message: str, retriable: bool = False):
        super().__init__(message)
        self.message = message
        self.retriable = retriable


class LLMProvider(ABC):
    def configure_tools(self, schemas):
        self.tool_schemas = schemas

    def usage_snapshot(self):
        return None

    @abstractmethod
    async def generate_structured(
        self,
        messages: list[dict[str, str]],
        response_schema: Type[T],
        temperature: float = 0.0
    ) -> tuple[T, dict[str, Any]]:
        """
        Generates structured output constrained by response_schema.
        Returns: (validated_instance, metadata)
        """
        pass


class GroqProvider(LLMProvider):
    _rate_state = {}
    def __init__(self, model=None, base_url=None, api_key=None, timeout=None,
                 max_output_tokens=None, transport=None):
        self.model = model or provider_model('groq', 'openai/gpt-oss-120b')
        self.base_url = (base_url or load_env_var("GROQ_BASE_URL", "https://api.groq.com/openai/v1")).rstrip("/")
        self.api_key = api_key or load_env_var("GROQ_KEY") or load_env_var("GROQ_API_KEY")
        self.timeout = float(timeout if timeout is not None else load_env_var("LLM_TIMEOUT_SECONDS", "20"))
        self.max_output_tokens = int(max_output_tokens or load_env_var("LLM_MAX_OUTPUT_TOKENS", "2048"))
        self.repair_output_tokens = min(self.max_output_tokens, 1024)
        self.reasoning_effort = load_env_var("LLM_REASONING_EFFORT", "medium")
        if self.reasoning_effort not in ("low", "medium", "high"):
            raise ValueError("Invalid reasoning effort")
        if not 0 < self.timeout <= 60 or not 128 <= self.max_output_tokens <= 4096:
            raise ValueError("Invalid provider bounds")
        self.transport = transport
        self.usage = {'requests': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'incomplete': False}

    @staticmethod
    def _clean_json_text(value):
        value = value.strip()
        if value.startswith("```"):
            value = "\n".join(value.splitlines()[1:])
            if value.endswith("```"):
                value = value[:-3].strip()
        return value

    @staticmethod
    def _reset_seconds(value):
        if value is None:
            return None
        text = str(value).strip()
        if re.fullmatch(r"\d+(?:\.\d+)?", text):
            seconds = float(text)
        elif re.fullmatch(r"(?:\d+(?:\.\d+)?(?:ms|s|m|h))+", text):
            units = {'ms': .001, 's': 1, 'm': 60, 'h': 3600}
            seconds = sum(float(amount) * units[unit] for amount, unit in
                          re.findall(r"(\d+(?:\.\d+)?)(ms|s|m|h)", text))
        else:
            return None
        return seconds if math.isfinite(seconds) else None

    @staticmethod
    def _duplicate_json_keys(content):
        duplicates = set()
        def inspect(pairs):
            seen = set()
            for key, _ in pairs:
                if key in seen:
                    duplicates.add(key)
                seen.add(key)
            return None
        json.loads(content, object_pairs_hook=inspect)
        return sorted(duplicates)

    async def generate_structured(self, messages, response_schema, temperature=0.0):
        if not self.api_key:
            raise ProviderError("GROQ_KEY is not configured")
        started = time.perf_counter()
        payload = {
            "model": self.model, "messages": list(messages), "temperature": temperature, "tool_choice": "none",
            "max_completion_tokens": self.max_output_tokens,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": response_schema.__name__, "schema": response_schema.model_json_schema(), "strict": False}},
        }
        if self.model.startswith("openai/gpt-oss-"):
            payload["reasoning_effort"] = self.reasoning_effort
        headers = {"Authorization": "Bearer " + self.api_key}
        attempts = transport_retries = repairs = 0
        rate_wait_seconds = 0.0
        rate_key = (self.base_url, self.model)
        prompt_tokens = completion_tokens = 0
        response_format = "json_schema"
        response_errors = []
        request_completion_caps = []
        scheduling_estimates = []
        cached_tokens = None
        try:
            async with asyncio.timeout(self.timeout * 2 + 62):
                async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport) as client:
                    while attempts < 4:
                        state = self._rate_state.get(rate_key) if self.transport is None else None
                        if state:
                            limit, remaining, stamp, *reset = state
                            elapsed = time.monotonic() - stamp
                            estimated = len(json.dumps(payload, separators=(",", ":"))) / 3.5 + payload['max_completion_tokens']
                            available = min(limit, remaining + elapsed * limit / 60)
                            scheduling_estimates.append({'estimated_tokens': estimated,
                                'token_limit': limit, 'exceeds_limit': estimated > limit})
                            # Cache treatment and tokenization are known only to the provider.
                            target = min(estimated, limit)
                            delay = max(0, (target - available) * 60 / limit)
                            if target > available and reset and reset[0] is not None:
                                delay = max(0, reset[0] - elapsed)
                            delay = min(60, delay)
                            if delay:
                                rate_wait_seconds += delay
                                await asyncio.sleep(delay)
                        attempts += 1
                        request_completion_caps.append(payload['max_completion_tokens'])
                        self.usage['requests'] += 1
                        try:
                            response = await client.post(self.base_url + "/chat/completions", headers=headers, json=payload)
                        except httpx.TimeoutException:
                            raise ProviderError("Groq request timed out", retriable=True) from None
                        except httpx.TransportError:
                            if transport_retries < 1:
                                transport_retries += 1
                                continue
                            raise ProviderError("Groq transport failed", retriable=True) from None
                        if self.transport is None and response.headers.get("x-ratelimit-limit-tokens"):
                            try:
                                limit = float(response.headers["x-ratelimit-limit-tokens"])
                                remaining = float(response.headers["x-ratelimit-remaining-tokens"])
                                if math.isfinite(limit) and limit > 0 and math.isfinite(remaining):
                                    self._rate_state[rate_key] = (limit, max(0, remaining), time.monotonic(),
                                        self._reset_seconds(response.headers.get('x-ratelimit-reset-tokens')))
                            except (KeyError, ValueError):
                                pass
                        if response.status_code in (429, 500, 502, 503, 504):
                            self.usage['incomplete'] = True
                            if response.status_code == 429:
                                try:
                                    message = response.json().get("error", {}).get("message", "")
                                except (ValueError, AttributeError):
                                    message = ""
                                if not isinstance(message, str):
                                    message = ""
                                if isinstance(message, str) and re.search(r"(?:tokens|requests) per day\s*\((?:TPD|RPD)\)", message, re.I):
                                    raise ProviderError("Groq daily quota exceeded; restore quota before starting another run")
                            if transport_retries < 1:
                                transport_retries += 1
                                delay = self._reset_seconds(response.headers.get('retry-after'))
                                if delay is None and response.status_code == 429:
                                    header = 'x-ratelimit-reset-requests' if re.search(r'requests per minute|\bRPM\b', message, re.I) else 'x-ratelimit-reset-tokens'
                                    delay = self._reset_seconds(response.headers.get(header))
                                delay = min(60, delay if delay is not None else .25)
                                rate_wait_seconds += delay
                                await asyncio.sleep(delay)
                                continue
                            raise ProviderError(f"Groq HTTP {response.status_code}; retry budget exhausted", retriable=True)
                        if response.status_code == 400 and response_format == "json_schema" and "json_schema" in response.text and any(term in response.text.lower() for term in ("unsupported", "not supported", "not available")):
                            response_format = "json_object"
                            payload["response_format"] = {"type": response_format}
                            continue
                        if response.status_code != 200:
                            self.usage['incomplete'] = True
                            try:
                                upstream_error = response.json().get("error", {})
                            except ValueError:
                                upstream_error = {}
                            error_code = upstream_error.get("code", "")
                            response_errors.append({"http_status": response.status_code, "code": error_code if isinstance(error_code, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", error_code) else "upstream_error"})
                            if response.status_code == 400 and error_code in ("json_validate_failed", "json_schema_validation_failed", "tool_use_failed") and repairs < 1:
                                repairs += 1
                                payload['max_completion_tokens'] = self.repair_output_tokens
                                failed = upstream_error.get("failed_generation", "")
                                payload["messages"] = list(messages) + [
                                    {"role": "assistant", "content": failed if isinstance(failed,str) and failed else "{}"},
                                    {"role": "user", "content": "The provider rejected the output contract. Do not issue native API function/tool calls; there are no API tools to invoke. Describe the selected action ONLY as JSON tool_name/tool_args when the schema allows them. Repair the output only; do not change the original goal. Respect action-dependent fields; inactive fields must be null or omitted. Return JSON matching: " + json.dumps(response_schema.model_json_schema())}]
                                continue
                            suffix = " (" + error_code + ")" if isinstance(error_code,str) and re.fullmatch(r"[A-Za-z0-9_-]{1,80}",error_code) else ""
                            raise ProviderError(f"Groq HTTP {response.status_code}" + suffix)
                        try:
                            data = response.json()
                            choice = data["choices"][0]
                            content = choice["message"]["content"]
                            if not isinstance(content, str) or choice.get("finish_reason") == "length":
                                raise ValueError("Missing or truncated content")
                        except (ValueError, KeyError, IndexError, TypeError):
                            self.usage['incomplete'] = True
                            raise ProviderError("Groq returned malformed or truncated chat content") from None
                        usage = data.get("usage") or {}
                        prompt_tokens += usage.get("prompt_tokens", 0)
                        completion_tokens += usage.get("completion_tokens", 0)
                        self.usage['prompt_tokens'] += usage.get('prompt_tokens', 0)
                        self.usage['completion_tokens'] += usage.get('completion_tokens', 0)
                        details = usage.get('prompt_tokens_details')
                        observed_cached = details.get('cached_tokens') if isinstance(details, dict) else None
                        if type(observed_cached) is int and observed_cached >= 0:
                            cached_tokens = (cached_tokens or 0) + observed_cached
                            self.usage['cached_tokens'] = self.usage.get('cached_tokens', 0) + observed_cached
                        if not usage:
                            self.usage['incomplete'] = True
                        errors = []
                        try:
                            clean_content = self._clean_json_text(content)
                            duplicates = self._duplicate_json_keys(clean_content)
                            if duplicates:
                                errors = [{'type': 'DUPLICATE_JSON_KEYS', 'keys': duplicates}]
                                response_errors.append({'code': 'DUPLICATE_JSON_KEYS', 'keys': duplicates})
                            else:
                                instance = response_schema.model_validate_json(clean_content)
                        except json.JSONDecodeError:
                            errors = [{'type': 'INVALID_JSON'}]
                            response_errors.append({'code': 'INVALID_JSON'})
                        except ValidationError as error:
                            errors = [{"field": list(item["loc"]), "type": item["type"], "message": item["msg"]} for item in error.errors(include_input=False, include_url=False)]
                            response_errors.append({"code": "application_schema_validation", "fields": [item['field'] for item in errors]})
                        if errors:
                            if repairs >= 1:
                                raise ProviderError("Groq output failed application schema validation after one repair: " + json.dumps(errors)) from None
                            repairs += 1
                            payload['max_completion_tokens'] = self.repair_output_tokens
                            payload["messages"] = list(messages) + [
                                {"role": "assistant", "content": content},
                                {"role": "user", "content": "Repair only the JSON contract. Validation errors: " + json.dumps(errors) + ". Match schema: " + json.dumps(response_schema.model_json_schema())}]
                            continue
                        metadata = {"provider": "groq", "model": self.model, "reasoning_effort": self.reasoning_effort,
                            "latency_seconds": round(time.perf_counter()-started, 3),
                            "prompt_eval_count": prompt_tokens, "eval_count": completion_tokens,
                            "attempts": attempts, "schema_repairs": repairs,
                            "transport_retries": transport_retries, "rate_wait_seconds": round(rate_wait_seconds,3), "response_format": response_format, "response_errors": response_errors,
                            'repair_max_completion_tokens': self.repair_output_tokens,
                            'request_completion_caps': request_completion_caps,
                            'scheduling_estimates': scheduling_estimates}
                        if cached_tokens is not None:
                            metadata['cached_tokens'] = cached_tokens
                        return instance, metadata
        except TimeoutError:
            self.usage['incomplete'] = True
            raise ProviderError("Groq overall request deadline exceeded", retriable=True) from None
        raise ProviderError("Groq request budget exhausted")

    def usage_snapshot(self):
        return dict(self.usage)


class GeminiProvider(LLMProvider):
    def __init__(self, model=None, base_url=None, api_key=None, timeout=None,
                 max_rpm=None, max_rpd=None, transport=None, rate_db_path=None):
        self.model = model or provider_model('gemini', 'gemini-3.5-flash-lite')
        self.base_url = (base_url or load_env_var('GEMINI_BASE_URL', 'https://generativelanguage.googleapis.com/v1beta')).rstrip('/')
        self.api_key = api_key or load_env_var('GEMINI_API_KEY')
        self.timeout = float(timeout if timeout is not None else load_env_var('LLM_TIMEOUT_SECONDS', '60'))
        self.max_rpm = int(max_rpm if max_rpm is not None else load_env_var('GEMINI_MAX_RPM', '12'))
        self.max_rpd = int(max_rpd if max_rpd is not None else load_env_var('GEMINI_MAX_RPD', '480'))
        self.max_output_tokens = int(load_env_var('LLM_MAX_OUTPUT_TOKENS', '2048'))
        if not 1 <= self.max_rpm <= 12 or not 1 <= self.max_rpd <= 498 or not 0 < self.timeout <= 60 or not 128 <= self.max_output_tokens <= 4096:
            raise ValueError('Invalid Gemini provider limits')
        self.transport = transport
        self.rate_db_path = Path(rate_db_path or load_env_var('OPERON_PROVIDER_LIMITS_DB', str(Path(__file__).resolve().parents[3] / 'data' / 'provider_limits.db')))
        self.rate_key = hashlib.sha256((self.base_url + '\0' + self.api_key).encode()).hexdigest()
        self.usage = {'requests': 0, 'prompt_tokens': 0, 'completion_tokens': 0,
                      'total_tokens': 0, 'thought_tokens': 0, 'incomplete': False, 'provider_errors': []}

    async def _acquire_rate_limit(self):
        waited = 0.0
        self.rate_db_path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            now = time.time()
            with sqlite3.connect(self.rate_db_path, timeout=5) as connection:
                connection.execute('CREATE TABLE IF NOT EXISTS provider_requests(account TEXT NOT NULL, timestamp REAL NOT NULL)')
                connection.execute('CREATE INDEX IF NOT EXISTS provider_request_time ON provider_requests(account,timestamp)')
                connection.execute('BEGIN IMMEDIATE')
                connection.execute('DELETE FROM provider_requests WHERE timestamp <= ?', (now - 86400,))
                count = connection.execute('SELECT COUNT(*) FROM provider_requests WHERE account=?', (self.rate_key,)).fetchone()[0]
                if count >= self.max_rpd:
                    raise ProviderError(f'Gemini daily quota limit ({self.max_rpd} requests) reached')
                timestamps = [row[0] for row in connection.execute('SELECT timestamp FROM provider_requests WHERE account=? AND timestamp>? ORDER BY timestamp', (self.rate_key, now - 60))]
                if len(timestamps) < self.max_rpm:
                    connection.execute('INSERT INTO provider_requests VALUES (?,?)', (self.rate_key, now))
                    return waited
                delay = max(0.1, timestamps[0] + 60.5 - now)
            await asyncio.sleep(min(60, delay))
            waited += min(60, delay)

    _clean_json_text = staticmethod(GroqProvider._clean_json_text)

    def _wire_schema(self, response_schema):
        from backend.app.agent.schemas import AgentDecision
        schema = response_schema.model_json_schema()
        if issubclass(response_schema, AgentDecision):
            properties = schema['properties']
            properties.setdefault('evidence', {'type': 'array', 'items': {'type': 'object'}})
            properties.setdefault('result', {'type': 'object'})
            properties.setdefault('replan', {'type': 'boolean'})
            names = [tool['name'] for tool in getattr(self, 'tool_schemas', [])]
            properties['tool_name'] = {'anyOf': [{'type': 'string', 'enum': names}, {'type': 'null'}]} if names else {'type': 'null'}
            properties['tool_args'] = {'type': 'object', 'description': 'Exact selected-tool arguments for act; {} for every non-act decision.'}
            schema['required'] = ['thought', 'action', 'tool_name', 'tool_args', 'clarification_question', 'failure_reason', 'evidence', 'result', 'replan']
        return schema

    def _argument_errors(self, tool, arguments):
        from jsonschema import Draft202012Validator
        return [{'path': list(error.path), 'validator': error.validator, 'message': error.message}
                for error in Draft202012Validator(tool['parameters']).iter_errors(arguments)]

    def _repair_tool(self, parsed):
        if not isinstance(parsed, dict) or parsed.get('action') != 'act':
            return None
        tool = next((tool for tool in getattr(self, 'tool_schemas', []) if tool['name'] == parsed.get('tool_name')), None)
        if tool is not None and ('tool_args' not in parsed or self._argument_errors(tool, parsed['tool_args'])):
            return tool
        return None

    def _validate(self, parsed, response_schema, wire_schema=None):
        from backend.app.agent.schemas import AgentDecision
        from jsonschema import Draft202012Validator
        if not issubclass(response_schema, AgentDecision):
            return response_schema.model_validate(parsed)
        errors = list(Draft202012Validator(wire_schema or self._wire_schema(response_schema)).iter_errors(parsed))
        if errors:
            raise ValueError('Decision wire contract: ' + '; '.join(error.message for error in errors))
        values = dict(parsed)
        if values['action'] == 'act':
            tool = next((tool for tool in getattr(self, 'tool_schemas', []) if tool['name'] == values['tool_name']), None)
            if tool is None:
                raise ValueError('Selected tool is not registered')
            errors = self._argument_errors(tool, values['tool_args'])
            if errors:
                raise ValueError('Selected tool arguments violate its registered contract: ' + json.dumps(errors))
        else:
            if values['tool_name'] is not None or values['tool_args'] != {}:
                raise ValueError('Non-act decisions require tool_name=null and tool_args={}')
            # Shared application contracts use None for an inactive tool; only the wire uses {}.
            values['tool_args'] = None
        for field, inactive in (('evidence', []), ('result', {}), ('replan', False)):
            if field not in response_schema.model_fields:
                if values[field] != inactive:
                    raise ValueError('This application Decision does not support ' + field)
                values.pop(field)
        return response_schema.model_validate(values)

    async def generate_structured(self, messages, response_schema, temperature=0.0):
        if not self.api_key:
            raise ProviderError('GEMINI_API_KEY is not configured')
        schema = self._wire_schema(response_schema)
        system = '\n\n'.join(message.get('content') or '' for message in messages if message.get('role') == 'system')
        inputs = [{key: value for key, value in message.items() if key in ('role', 'content', 'tool_calls', 'tool_call_id')}
                  for message in messages if message.get('role') != 'system']
        payload = {'model': self.model, 'system_instruction': system, 'input': json.dumps({'messages': inputs}),
                   'response_format': {'type': 'text', 'mime_type': 'application/json', 'schema': schema},
                   'generation_config': {'max_output_tokens': self.max_output_tokens, 'tool_choice': 'none'}, 'store': False}
        from backend.app.agent.schemas import AgentDecision
        contract = None
        if issubclass(response_schema, AgentDecision):
            contract = {'required_fields': schema['required'],
                        'act': 'Return the COMPLETE Decision, including the selected registered tool_name and its exact tool_args.',
                        'non_act': 'Return the COMPLETE Decision with tool_name=null and tool_args={}. need_clarification requires clarification_question; fail requires failure_reason.'}
            payload['input'] = json.dumps({'messages': inputs, 'response_contract': contract})
        headers = {'x-goog-api-key': self.api_key, 'Content-Type': 'application/json'}
        started = time.perf_counter()
        rate_wait = 0.0
        prompt_tokens = completion_tokens = total_tokens = thought_tokens = 0
        response_errors = []
        for attempt in range(2):
            rate_wait += await self._acquire_rate_limit()
            self.usage['requests'] += 1
            try:
                async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport) as client:
                    response = await client.post(self.base_url + '/interactions', headers=headers, json=payload)
            except httpx.TransportError:
                self.usage['incomplete'] = True
                self.usage['provider_errors'].append({'code': 'transport_error'})
                raise ProviderError('Gemini transport failed or request timed out', retriable=True) from None
            if response.status_code != 200:
                self.usage['incomplete'] = True
                error = {'http_status': response.status_code}
                self.usage['provider_errors'].append(error)
                raise ProviderError(f'Gemini HTTP {response.status_code}', retriable=response.status_code in (429, 500, 502, 503, 504))
            try:
                data = response.json()
                if not isinstance(data, dict):
                    raise ValueError('Unexpected response shape')
            except ValueError:
                self.usage['incomplete'] = True
                raise ProviderError('Gemini returned a malformed interaction') from None
            usage = data.get('usage') or {}
            if not isinstance(usage, dict) or not usage or any(not isinstance(usage.get(key, 0), int) or isinstance(usage.get(key, 0), bool) or usage.get(key, 0) < 0
                                for key in ('total_input_tokens', 'total_output_tokens', 'total_tokens', 'total_thought_tokens')) or any(key not in usage for key in ('total_input_tokens', 'total_output_tokens', 'total_tokens')):
                self.usage['incomplete'] = True
                usage = {}
            p, c, total, thought = (usage.get(key, 0) for key in ('total_input_tokens', 'total_output_tokens', 'total_tokens', 'total_thought_tokens'))
            prompt_tokens += p
            completion_tokens += c
            total_tokens += total
            thought_tokens += thought
            for key, value in (('prompt_tokens', p), ('completion_tokens', c), ('total_tokens', total), ('thought_tokens', thought)):
                self.usage[key] += value
            if data.get('status') != 'completed':
                self.usage['provider_errors'].append({'code': 'interaction_not_completed'})
                raise ProviderError('Gemini interaction did not complete')
            text = ''
            parsed = None
            try:
                parts = [content.get('text', '') for step in data.get('steps', []) if step.get('type') == 'model_output'
                         for content in step.get('content', []) if content.get('type') == 'text']
                text = ''.join(parts)
                if not text:
                    raise ValueError('Missing model text')
                parsed = json.loads(self._clean_json_text(text))
                instance = self._validate(parsed, response_schema, payload['response_format']['schema'])
            except (ValueError, TypeError, AttributeError) as error:
                if isinstance(error, ValidationError):
                    reason = json.dumps(error.errors(include_input=False, include_url=False, include_context=False))
                elif isinstance(error, json.JSONDecodeError):
                    reason = 'Output is not valid JSON'
                else:
                    reason = str(error) if isinstance(error, ValueError) else 'Malformed output structure'
                response_errors.append({'code': 'schema_validation'})
                if attempt:
                    self.usage['provider_errors'].extend(response_errors)
                    raise ProviderError('Gemini output failed schema/tool validation after one repair') from None
                repair = {'instruction': 'Return the COMPLETE corrected JSON object matching the response schema.', 'validation_error': reason}
                tool = self._repair_tool(parsed) if contract else None
                if tool is not None:
                    from copy import deepcopy
                    narrowed = deepcopy(schema)
                    narrowed['properties']['action'] = {'type': 'string', 'enum': ['act']}
                    narrowed['properties']['tool_name'] = {'type': 'string', 'enum': [tool['name']]}
                    narrowed['properties']['tool_args'] = deepcopy(tool['parameters'])
                    payload['response_format']['schema'] = narrowed
                    repair.update(selected_tool=tool['name'], tool_args_schema=tool['parameters'])
                    repair['instruction'] += ' Keep action=act and tool_name=' + tool['name'] + '; do not change the selected tool.'
                payload['input'] = json.dumps({'messages': inputs, 'response_contract': contract, 'previous_output': text, 'repair': repair})
                continue
            return instance, {'provider': 'gemini', 'model': self.model,
                'latency_seconds': round(time.perf_counter() - started, 3), 'prompt_eval_count': prompt_tokens,
                'eval_count': completion_tokens, 'total_tokens': total_tokens, 'thought_tokens': thought_tokens,
                'attempts': attempt + 1, 'schema_repairs': attempt, 'response_errors': response_errors,
                'rate_wait_seconds': round(rate_wait, 3)}

    def usage_snapshot(self):
        return json.loads(json.dumps(self.usage))


def get_default_provider(**kwargs) -> LLMProvider:
    provider_name = load_env_var('LLM_PROVIDER', 'gemini').lower()
    if provider_name == 'groq':
        return GroqProvider(**kwargs)
    if provider_name == 'gemini':
        return GeminiProvider(**kwargs)
    raise ValueError('LLM_PROVIDER must be gemini or groq')


class OllamaProvider(LLMProvider):
    """Compatibility wrapper for historical callers of the provider interface."""
    def __init__(self, **kwargs):
        self._provider = get_default_provider(**kwargs)

    @property
    def model(self):
        return self._provider.model

    def configure_tools(self, schemas):
        self._provider.configure_tools(schemas)

    async def generate_structured(self, messages, response_schema, temperature=0.0):
        return await self._provider.generate_structured(messages, response_schema, temperature)

    def usage_snapshot(self):
        return self._provider.usage_snapshot()


class FakeProvider(LLMProvider):
    """Deterministic scripted provider for testing agent state transitions without model inference."""
    def __init__(self, responses: list[Any] | None = None):
        self.responses: list[Any] = list(responses or [])
        self.call_history: list[list[dict[str, str]]] = []

    def queue_response(self, response: Any) -> None:
        self.responses.append(response)

    async def generate_structured(
        self,
        messages: list[dict[str, str]],
        response_schema: Type[T],
        temperature: float = 0.0
    ) -> tuple[T, dict[str, Any]]:
        self.call_history.append(messages)
        if not self.responses:
            raise ProviderError("FakeProvider ran out of scripted responses")
        
        next_resp = self.responses.pop(0)
        if isinstance(next_resp, response_schema):
            return next_resp, {"model": "fake", "latency_seconds": 0.001}
        elif isinstance(next_resp, dict):
            return response_schema.model_validate(next_resp), {"model": "fake", "latency_seconds": 0.001}
        else:
            return response_schema.model_validate_json(str(next_resp)), {"model": "fake", "latency_seconds": 0.001}
