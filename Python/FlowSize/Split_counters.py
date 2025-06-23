# Define a mapping of feature count to output file names.
import sys
if sys.argv[1] == "1":
    output_files = {4: "equinix-nyc1_output_4.csv",
                    5: "equinix-nyc1_output_5.csv",
                    6: "equinix-nyc1_output_6.csv",
                    7: "equinix-nyc1_output_7.csv",
                    8: "equinix-nyc1_output_8.csv"}
    raw_path = "/home/takatsukiaa/ML-Sketch/Python/FlowSize/training_flows.csv"

else:
    output_files = {4: "equinix-nyc1_output_4_2.csv",
                    5: "equinix-nyc1_output_5_2.csv",
                    6: "equinix-nyc1_output_6_2.csv",
                    7: "equinix-nyc1_output_7_2.csv",
                    8: "equinix-nyc1_output_8_2.csv"}
    raw_path = "/home/takatsukiaa/ML-Sketch/Python/FlowSize/testing_flows.csv"

# Open the output files for writing.
file_handles = {count: open(fname, "w") for count, fname in output_files.items()}

# Process the input file line by line.
with open(raw_path, "r") as infile:
    for line in infile:
        # Split the line by whitespace.
        tokens = line.strip().split(',')
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