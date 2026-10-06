"""Generic aggregation/comparison checks independent of generated analysis code."""
import math
from typing import Literal
from pydantic import Field
from backend.app.agent.schemas import StrictModel
from backend.app.capabilities.python.profile import load_dataset


class CalculationContract(StrictModel):
    document_id: str
    group_column: str
    value_column: str
    aggregate: Literal['sum', 'mean', 'count', 'min', 'max'] = 'sum'
    period_column: str | None = None
    baseline_period: str | None = None
    current_period: str | None = None
    measure: Literal['value', 'difference', 'percent_change']
    convention: Literal['current_minus_baseline', 'baseline_minus_current'] = 'current_minus_baseline'
    selection: Literal['min', 'max']
    group_metric: str = Field(description='Result metrics key naming the selected group')
    value_metric: str = Field(description='Result metrics key holding the calculated measure')
    unsupported_requirements: list[str] = Field(default_factory=list)


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
