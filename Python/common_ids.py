import re

# Read the contents of both files
with open("/home/takatsukiaa/ML-Sketch/Python/predict_heap.txt", "r") as f:
    predict_text = f.read()

with open("/home/takatsukiaa/ML-Sketch/Python/actual_heap.txt", "r") as f:
    actual_text = f.read()

# Regular expression to extract IDs
id_pattern = r"ID: ([A-Z0-9]+)"

# Find all IDs in both files
predict_ids = set(re.findall(id_pattern, predict_text))
actual_ids = set(re.findall(id_pattern, actual_text))

# Find intersection
common_ids = sorted(predict_ids & actual_ids)
size = 0
for id in common_ids:
    print(id)
    size = size + 1
print(size)
import pandas as pd