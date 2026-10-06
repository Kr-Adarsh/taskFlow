"""Bounded admission of model-written, read-only verification contracts."""
import json
from copy import deepcopy
from pydantic import ConfigDict, create_model


def required_response_schema(base, *, branches=None, properties=None):
    """Constrain generation without changing stored models or supplying answer facts."""
    def constrain(schema):
        schema['required'] = list(schema['properties'])
        for field in schema['properties'].values():
            field.pop('default', None)
        for name, constraint in (properties or {}).items():
            description = schema['properties'][name].get('description')
            schema['properties'][name] = deepcopy(constraint)
            if description:
                schema['properties'][name]['description'] = description
        if branches:
            schema['anyOf'] = deepcopy(branches)
    return create_model(base.__name__, __base__=base,
                        __config__=ConfigDict(json_schema_extra=constrain))


class ContractInterpretationError(ValueError):
    def __init__(self, attempts, contract):
        self.attempts = attempts
        self.contract = contract
        super().__init__('Verification contract remained invalid after one interpretation repair: ' + attempts[-1]['error'])


async def interpret_contract(provider, messages, schema, validate):
    """Keep rejected interpretations as evidence; never execute or silently fill them."""
    attempts = []
    request = list(messages)
    for attempt in range(2):
        contract, _ = await provider.generate_structured(request, schema)
        record = {'contract': contract.model_dump(), 'repair': attempt == 1}
        try:
            validate(contract)
            # Provider constraints are guidance as well as a contract. Check them locally.
            from jsonschema import Draft202012Validator
            issues = sorted(Draft202012Validator(schema.model_json_schema()).iter_errors(record['contract']),
                            key=lambda error: str(error.path))
            if issues:
                raise ValueError('Generated verification contract violates its required shape: ' +
                                 '; '.join(error.message for error in issues[:4]))
        except ValueError as error:
            record.update(admitted=False, error=str(error))
            attempts.append(record)
            if attempt == 0:
                request = [*messages, {'role': 'user', 'content': json.dumps({
                    'interpretation_repair': {
                        'previous_contract': record['contract'], 'admission_error': str(error),
                        'instruction': 'Reinterpret the unchanged original objective using the declared checking capabilities. '
                                       'Preserve every explicit identity, condition, priority, period, direction and unit. '
                                       'Correct omitted or contradictory fields. If a requirement truly cannot be represented, '
                                       'retain it as unsupported. Do not infer success or expected values from execution. '
                                       'Return only the corrected schema. This is the only interpretation repair.',
                    }})}]
                continue
            raise ContractInterpretationError(attempts, contract) from error
        record['admitted'] = True
        attempts.append(record)
        return contract, attempts
