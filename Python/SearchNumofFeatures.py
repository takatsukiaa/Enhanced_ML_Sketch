import pandas as pd

# Load the CSV file
file_path = "flows.csv"  # Replace with your CSV file path
data = pd.read_csv(file_path, header=None)

# Check the first column
first_column = data.iloc[:, 0]  # Extract the first column

# Values to search for
values_to_search = [3, 4, 5, 6]

# Count occurrences
counts = first_column.value_counts()

# Filter counts for the specific values
filtered_counts = counts[counts.index.isin(values_to_search)]

# Print the counts
print("Occurrences of each value:")
for value in values_to_search:
    print(f"Value {value}: {filtered_counts.get(value, 0)} times")