import pandas as pd

# Load the CSV
df = pd.read_csv(inputs["sales.csv"])

# Ensure month column is string for filtering
df['month'] = df['month'].astype(str)

# Keep only the two months of interest
months = ['2026-08', '2026-09']
filtered = df[df['month'].isin(months)]

# Aggregate revenue per region per month
pivot = filtered.pivot_table(index='region', columns='month', values='revenue', aggfunc='sum', fill_value=0)

# Compute absolute decline between the months
pivot['decline'] = (pivot['2026-08'] - pivot['2026-09']).abs()

# Identify region with maximum decline
max_region = pivot['decline'].idxmax()
max_decline = pivot.loc[max_region, 'decline']

# Prepare result adhering to contract
result = {
    "summary": f"Region with the largest absolute revenue decline between 2026-08 and 2026-09 is {max_region} with a decline of {max_decline}.",
    "metrics": {
        "region_with_max_decline": max_region,
        "max_absolute_decline": max_decline
    }
}