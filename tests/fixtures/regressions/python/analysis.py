import pandas as pd

df = pd.read_csv(inputs['sales.csv'])
# Aggregate revenue per region and month
agg = df.groupby(['region', 'month'])['revenue'].sum().reset_index()
# Pivot to have months as columns
pivot = agg.pivot(index='region', columns='month', values='revenue').fillna(0)
# Calculate absolute decline between the two months
pivot['decline'] = (pivot['2026-08'] - pivot['2026-09']).abs()
# Identify region with maximum decline
max_region = pivot['decline'].idxmax()
max_decline = pivot.loc[max_region, 'decline']
result = {
    "region": max_region,
    "decline": float(max_decline)
}