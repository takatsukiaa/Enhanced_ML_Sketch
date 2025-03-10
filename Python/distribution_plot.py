import pandas as pd
import matplotlib.pyplot as plt

# Load the file
file_path = "/home/takatsukiaa/ML-Sketch/Python/equinix-chicago1_flows copy.csv"
df = pd.read_csv(file_path, header=None)

# Extract the second column as X-axis values
x_values = df[1]

# Count occurrences of each unique value in X-axis
flow_counts = x_values.value_counts().sort_index()

# Get the maximum Y value
max_y_value = max(flow_counts.values)

# Plot with log scale and max value marked with a dot
plt.figure(figsize=(12, 6))
plt.bar(flow_counts.index, flow_counts.values, width=1.5, log=True)

# Set X-axis limit to include flow sizes up to 4096
plt.xlim(min(flow_counts.index), 5000)
plt.ylim(1, max_y_value)

# Annotate vertical lines for key flow sizes
annotate_points = [512, 1024, 2048, 4096]
for point in annotate_points:
    plt.axvline(x=point, color='red', linestyle='--', linewidth=1)
    plt.text(point, max_y_value / 10, f"{point}", ha='center', fontsize=10, color='red', rotation=90)

# Mark max value with a dot
max_x_value = flow_counts.idxmax()
plt.scatter(max_x_value, max_y_value, color='blue', s=100, label="Max Value")
plt.text(max_x_value, max_y_value, f"Max: {max_y_value}", ha='right', fontsize=10, color='blue')

plt.xlabel("Flow Size")
plt.ylabel("Number of Flows (Log Scale)")
plt.title("Flow Size Distribution")

plt.grid(False)
plt.legend()
plt.show()
