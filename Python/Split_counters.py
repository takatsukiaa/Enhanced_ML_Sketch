# Define a mapping of feature count to output file names.
output_files = {4: "output_4.csv",
                5: "output_5.csv",
                6: "output_6.csv",
                7: "output_7.csv",
                8: "output_8.csv"}

# Open the output files for writing.
file_handles = {count: open(fname, "w") for count, fname in output_files.items()}

# Process the input file line by line.
with open("/home/takatsukiaa/ML-Sketch/Python/flows.csv", "r") as infile:
    for line in infile:
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
        
        # Write the line to the corresponding file if the feature_count is within our expected range.
        if feature_count in file_handles:
            file_handles[feature_count].write(line)

# Close all output files.
for f in file_handles.values():
    f.close()