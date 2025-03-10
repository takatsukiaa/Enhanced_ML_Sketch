import pandas as pd
# Values to search for
values_to_search = [4, 5, 6, 7, 8]
# Load the CSV file
counts = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
file_path = "/home/takatsukiaa/ML-Sketch/Python/equinix-chicago1_flows.csv"  # Replace with your CSV file path
with open(file_path,"r") as file:
    for line in file:
            # Split the line by whitespace.
            tokens = line.strip().split()
            if not tokens:  # Skip empty lines.
                continue
            try:
                # The first token should be the feature count.
                feature_count = int(tokens[0])
            except ValueError:
                # If conversion fails, skip this line.
                continue
            if feature_count in values_to_search:
                 counts[feature_count] += 1
            else:
                continue

# Filter counts for the specific values
# Print the counts
print("Occurrences of each value:")
for value in values_to_search:
    print(f"Value {value}: {counts[value]} times")