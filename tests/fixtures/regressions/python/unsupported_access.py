import pandas as pd
df = pd.read_csv('sales.csv')
aug = df[df['month'] == '2026-08']
sep = df[df['month'] == '2026-09']
aug_sum = aug.groupby('region')['revenue'].sum()
sep_sum = sep.groupby('region')['revenue'].sum()
regions = set(aug_sum.index).union(sep_sum.index)
declines = {}
for r in regions:
    rev_aug = aug_sum.get(r, 0)
    rev_sep = sep_sum.get(r, 0)
    declines[r] = abs(rev_aug - rev_sep)
max_region = max(declines, key=declines.get)
max_decline = declines[max_region]
result = {
    "region_with_largest_decline": max_region,
    "decline_amount": max_decline,
    "declines_by_region": declines
}