import os
def split():
    # Define a mapping of feature count to output file names.
    output_files = {4: "/home/takatsukiaa/ML-Sketch/Python/TopK/equinix-chicago1_output_4.csv",
                    5: "/home/takatsukiaa/ML-Sketch/Python/TopK/equinix-chicago1_output_5.csv",
                    6: "/home/takatsukiaa/ML-Sketch/Python/TopK/equinix-chicago1_output_6.csv",
                    7: "/home/takatsukiaa/ML-Sketch/Python/TopK/equinix-chicago1_output_7.csv",
                    8: "/home/takatsukiaa/ML-Sketch/Python/TopK/equinix-chicago1_output_8.csv"}

    # Open the output files for writing.
    file_handles = {count: open(fname, "w") for count, fname in output_files.items()}
    # Process the input file line by line.
    try:
        with open("/home/takatsukiaa/ML-Sketch/Python/TopK/training_flows.csv", "r") as infile:
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
    except:
        os.remove("/home/takatsukiaa/ML-Sketch/Python/TopK/equinix-chicago1_output_4.csv")
        os.remove("/home/takatsukiaa/ML-Sketch/Python/TopK/equinix-chicago1_output_5.csv")
        os.remove("/home/takatsukiaa/ML-Sketch/Python/TopK/equinix-chicago1_output_6.csv")
        os.remove("/home/takatsukiaa/ML-Sketch/Python/TopK/equinix-chicago1_output_7.csv")
        os.remove("/home/takatsukiaa/ML-Sketch/Python/TopK/equinix-chicago1_output_8.csv")
        return 1
    return 0