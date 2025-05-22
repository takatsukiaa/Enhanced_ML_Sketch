import matplotlib.pyplot as plt
import numpy as np

# Bar heights
group_labels = ['CM', 'CU', 'ML_CM', 'ML_CU']  # 4 groups
bar_values_group1 = [1.24, 2.05]        # Group C
bar_values_group2 = [3.48, 2.12]        # Group D

# Positions for bars
x = np.arange(len(group_labels))  # [0, 1, 2, 3]
bar_width = 0.35

fig, ax = plt.subplots()

# Single bars for A and B
bar_a = ax.bar(x[0], 543.61, width=bar_width, color='blue')
bar_b = ax.bar(x[1], 306.81, width=bar_width, color='green')

# Grouped bars for C and D
bar_c1 = ax.bar(x[2] - bar_width/2, bar_values_group1[0], width=bar_width, label='Training', color='orange')
bar_c2 = ax.bar(x[2] + bar_width/2, bar_values_group1[1], width=bar_width, label='Testing', color='red')

bar_d1 = ax.bar(x[3] - bar_width/2, bar_values_group2[0], width=bar_width, label='Training', color='purple')
bar_d2 = ax.bar(x[3] + bar_width/2, bar_values_group2[1], width=bar_width, label='Testing', color='pink')

ax.bar_label(bar_a)
ax.bar_label(bar_b)
ax.bar_label(bar_c1)
ax.bar_label(bar_c2)
ax.bar_label(bar_d1)
ax.bar_label(bar_d2)

# Formatting
ax.set_xticks(x)
ax.set_xticklabels(group_labels)
ax.set_ylabel('MRE')
ax.set_title('MRE of Different Approach')
ax.legend()
plt.tight_layout()
plt.show()