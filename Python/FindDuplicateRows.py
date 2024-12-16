import pandas as pd

# Load your dataset
df = pd.read_csv('convert.csv', header=None)  # Use header=None if the first row is not a column header

# Ensure all values are treated as strings (avoids type issues)
df = df.astype(str)

# Sort the values in each row and create a new column for sorted rows
df['sorted_row'] = df.apply(lambda row: tuple(sorted(row)), axis=1)

# Identify duplicate rows based on the sorted row values
duplicate_rows = df[df.duplicated('sorted_row', keep=False)]

# Drop the helper column
duplicate_rows = duplicate_rows.drop(columns=['sorted_row'])

# Save duplicates to a CSV file
duplicate_rows.to_csv('duplicates.csv', index=False, header=False)  # Use header=False if no column names

print("Duplicates have been saved to 'duplicates.csv'")
