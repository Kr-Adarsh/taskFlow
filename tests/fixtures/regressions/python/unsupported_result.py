import pandas as pd, os
df = pd.read_csv(inputs["sales.csv"])
# Aggregate revenue per region and month
pivot = df.groupby(["region", "month"])['revenue'].sum().unstack(fill_value=0)
# Compute absolute decline between August and September 2026
pivot['decline'] = (pivot['2026-08'] - pivot['2026-09']).abs()
# Identify region with the maximum decline
max_region = pivot['decline'].idxmax()
max_decline = pivot.loc[max_region, 'decline']
result = {"region": max_region, "decline": int(max_decline)}