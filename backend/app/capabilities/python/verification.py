"""Generic aggregation/comparison checks independent of generated analysis code."""
import math
import re
from typing import Literal
from pydantic import Field
from backend.app.agent.schemas import StrictModel
from backend.app.agent.interpretation import required_response_schema
from backend.app.capabilities.python.profile import load_dataset


class CalculationContract(StrictModel):
    document_id: str
    group_column: str
    value_column: str
    aggregate: Literal['sum', 'mean', 'count', 'min', 'max'] = 'sum'
    period_column: str | None = None
    baseline_period: str | None = None
    current_period: str | None = None
    measure: Literal['value', 'difference', 'percent_change'] = Field(description="value for an aggregation; difference for a change in original numeric units, including an absolute revenue decline; percent_change for a relative percentage. Largest decline in original units is a supported difference, not an unsupported absolute-value operation.")
    convention: Literal['current_minus_baseline', 'baseline_minus_current'] = Field(default='current_minus_baseline', description="For a decline reported as a positive amount, use baseline_minus_current with max selection. Preserve the objective's direction and units; a largest absolute change across both increases and decreases is a different requirement.")
    selection: Literal['min', 'max']
    group_metric: str = Field(description='Result metrics key naming the selected group')
    value_metric: str = Field(description='Result metrics key holding the calculated measure')
    unsupported_requirements: list[str] = Field(default_factory=list, description='Only original requirements the declared aggregation/comparison cannot represent. A positive unit-valued decline is supported as baseline_minus_current difference; it is not an absolute-change requirement across increases and decreases.')


def calculation_schema(objective, input_ids, metric_keys):
    text = {'type': 'string', 'minLength': 1}
    properties = {'document_id': {'type': 'string', 'enum': sorted(input_ids)},
                  'group_metric': {'type': 'string', 'enum': sorted(metric_keys)},
                  'value_metric': {'type': 'string', 'enum': sorted(metric_keys)}}
    periods = re.search(r'\b(?:from|between)\s+(\d{4}-\d{2}(?:-\d{2})?)\s+(?:to|and)\s+(\d{4}-\d{2}(?:-\d{2})?)\b', objective, re.I)
    if periods:
        properties['baseline_period'] = {'type': 'string', 'enum': [periods[1]]}
        properties['current_period'] = {'type': 'string', 'enum': [periods[2]]}
    decline = re.search(r'\b(?:largest|greatest|biggest|maximum)\b[^.!?;]{0,60}\b(?:decline|decrease|drop)\b', objective, re.I)
    signed_or_percent = re.search(r'\b(?:signed|negative|percent|percentage)\b|%', objective, re.I)
    if decline and not signed_or_percent:
        properties.update(measure={'type': 'string', 'enum': ['difference']},
                          convention={'type': 'string', 'enum': ['baseline_minus_current']},
                          selection={'type': 'string', 'enum': ['max']})
    return required_response_schema(CalculationContract, properties=properties, branches=[
        {'properties': {'measure': {'enum': ['value']}}},
        {'properties': {'measure': {'enum': ['difference', 'percent_change']},
                        'period_column': text, 'baseline_period': text, 'current_period': text},
         'required': ['period_column', 'baseline_period', 'current_period']},
    ])


def validate_calculation_contract(contract, objective, profiles, input_ids, metric_keys):
    if contract.unsupported_requirements:
        raise ValueError('Unsupported calculation requirements: ' + '; '.join(contract.unsupported_requirements))
    if contract.document_id not in input_ids or contract.document_id not in profiles:
        raise ValueError('Verification input must be one of the profiled execution inputs')
    if contract.group_metric == contract.value_metric or not {contract.group_metric, contract.value_metric} <= set(metric_keys):
        raise ValueError('Distinct group and value metric keys must belong to the execution output schema')
    columns = set(profiles[contract.document_id]['columns'])
    required = {contract.group_column, contract.value_column}
    if contract.measure != 'value':
        if not contract.period_column or not contract.baseline_period or not contract.current_period or contract.baseline_period == contract.current_period:
            raise ValueError('A comparison requires explicit, distinct baseline/current periods and a period column')
        required.add(contract.period_column)
    if not required <= columns or contract.group_column == contract.value_column:
        raise ValueError('Calculation columns must match the independent dataset schema')
    periods = re.search(r'\b(?:from|between)\s+(\d{4}-\d{2}(?:-\d{2})?)\s+(?:to|and)\s+(\d{4}-\d{2}(?:-\d{2})?)\b', objective, re.I)
    if periods and (contract.baseline_period, contract.current_period) != periods.groups():
        raise ValueError('The comparison must preserve the explicitly ordered original periods')
    # A unit-valued decline is directional. An increase cannot win merely by magnitude.
    decline = re.search(r'\b(?:largest|greatest|biggest|maximum)\b[^.!?;]{0,60}\b(?:decline|decrease|drop)\b', objective, re.I)
    signed_or_percent = re.search(r'\b(?:signed|negative|percent|percentage)\b|%', objective, re.I)
    if decline and not signed_or_percent:
        if (contract.measure, contract.convention, contract.selection) != ('difference', 'baseline_minus_current', 'max'):
            raise ValueError('A largest decline in original units requires difference, baseline_minus_current and max selection; preserve the original direction')


def calculate(contract):
    _, frame = load_dataset(contract.document_id)
    columns = [contract.group_column, contract.value_column]
    if contract.period_column:
        columns.append(contract.period_column)
    if any(column not in frame.columns for column in columns) or frame[columns].isna().any().any():
        raise ValueError('Required calculation columns are missing or contain unresolved null values')
    if contract.group_column == contract.value_column:
        raise ValueError('Group and value columns must differ')
    import pandas as pd
    frame[contract.value_column] = pd.to_numeric(frame[contract.value_column], errors='raise')
    if not all(math.isfinite(float(value)) for value in frame[contract.value_column]):
        raise ValueError('Calculation requires finite numeric values')
    if contract.measure == 'value':
        values = frame.groupby(contract.group_column)[contract.value_column].agg(contract.aggregate)
    else:
        if not contract.period_column or not contract.baseline_period or not contract.current_period or contract.baseline_period == contract.current_period:
            raise ValueError('A comparison needs distinct, explicit baseline/current periods')
        periods = frame[contract.period_column].astype(str)
        baseline = frame[periods == contract.baseline_period].groupby(contract.group_column)[contract.value_column].agg(contract.aggregate)
        current = frame[periods == contract.current_period].groupby(contract.group_column)[contract.value_column].agg(contract.aggregate)
        if baseline.empty or current.empty or set(baseline.index) != set(current.index):
            raise ValueError('Each group must have observations in both requested periods')
        values = current - baseline
        if contract.measure == 'percent_change':
            if (baseline == 0).any():
                raise ValueError('Percent change is undefined for a zero baseline')
            values = values / baseline * 100
        if contract.convention == 'baseline_minus_current':
            values = -values
    if values.empty:
        raise ValueError('No matching observations')
    extreme = values.min() if contract.selection == 'min' else values.max()
    winners = values[values == extreme]
    if len(winners) != 1:
        raise ValueError('The requested selection is tied; a unique answer is unresolved')
    return {'group': str(winners.index[0]), 'value': float(extreme), 'full_dataset_rows': len(frame),
            'groups_compared': len(values), 'group_values_preview': {str(key): float(value) for key, value in values.head(10).items()}}
