import pandas as pd

# Load the CSV file
file_path = 'hash_values3.csv'  # Replace with your file path
data = pd.read_csv(file_path, header=None)

# Identify duplicate rows
duplicate_rows = data[data.duplicated(keep=False)]

# Display duplicate rows
if duplicate_rows.empty:
    print("No duplicate rows found.")
else:
    print("Duplicate rows:")
    print(duplicate_rows)

# Optionally, save duplicate rows to a new CSV file
duplicate_rows.to_csv('duplicates.csv', index=False)