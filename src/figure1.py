import matplotlib.pyplot as plt
import numpy as np



gap = 100
#Open npz file
limit=30000
#tree-attention
data1 = np.load('../data/vocab2_seed/1layer_n25_tree_seed61.npz')



time_stamps1 = data1['timestamps'][:limit:gap]
accuracy1 = data1['accuracy'][:limit:gap]
loss1 = data1['loss'][:limit:gap]
num_epochs1 = data1['num_epochs']
num_layers1 = data1['num_layers']

#1 layer self-attention
data2 = np.load('../data/vocab2_seed/1layer_n25_seed61.npz')

time_stamps2 = data2['timestamps'][:limit:gap]
accuracy2 = data2['accuracy'][:limit:gap]
loss2 = data2['loss'][:limit:gap]
num_epochs2 = data2['num_epochs']
num_layers2 = data2['num_layers']

#2 layer self-attention
data3 = np.load('../data/vocab2_seed/2layer_n25_seed61.npz')

time_stamps3 = data3['timestamps'][:limit:gap]
accuracy3 = data3['accuracy'][:limit:gap]
loss3 = data3['loss'][:limit:gap]
num_epochs3 = data3['num_epochs']
num_layers3 = data3['num_layers']

epochs = np.arange(1, limit + 1, gap)
cumulative_time1 = np.cumsum(time_stamps1)
cumulative_time2 = np.cumsum(time_stamps2)
cumulative_time3 = np.cumsum(time_stamps3)

# Calculate and print total running times in minutes
total_time1_minutes = cumulative_time1[-1] / 60
total_time2_minutes = cumulative_time2[-1] / 60
total_time3_minutes = cumulative_time3[-1] / 60

print("Total Running Times:")
print(f"Tree-attention: {total_time1_minutes:.2f} minutes")
print(f"1-layer self-attention: {total_time2_minutes:.2f} minutes")
print(f"2-layer self-attention: {total_time3_minutes:.2f} minutes")

# Define format strings and labels for consistent styling
fmts = ['r', 'g', 'b']  # red triangles, green inverted triangles, blue circles
labels = ['Tree-attention', '1-layer self-attention', '2-layer self-attention']

# Plot only Accuracy vs Epochs
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# Accuracy vs Epochs
ax.plot(epochs, accuracy1, fmts[0], linewidth=2, markersize=4, markevery=20)
ax.plot(epochs, accuracy2, fmts[1], linewidth=2, markersize=4, markevery=20)
ax.plot(epochs, accuracy3, fmts[2], linewidth=2, markersize=4, markevery=20)
ax.set_xlabel('Epochs')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy vs Epochs')
ax.grid(True, alpha=0.3)

# # Plot 3: Total Time vs Epochs
# axes[1, 0].plot(epochs, cumulative_time1, fmts[0], linewidth=2, markersize=4, markevery=20)
# axes[1, 0].plot(epochs, cumulative_time2, fmts[1], linewidth=2, markersize=4, markevery=20)
# axes[1, 0].plot(epochs, cumulative_time3, fmts[2], linewidth=2, markersize=4, markevery=20)
# axes[1, 0].set_xlabel('Epochs')
# axes[1, 0].set_ylabel('Total Time (s)')
# axes[1, 0].set_title('Total Time vs Epochs')
# axes[1, 0].grid(True, alpha=0.3)

# # Plot 4: Accuracy vs Total Time
# axes[1, 1].plot(cumulative_time1, accuracy1, fmts[0], linewidth=2, markersize=4, markevery=20)
# axes[1, 1].plot(cumulative_time2, accuracy2, fmts[1], linewidth=2, markersize=4, markevery=20)
# axes[1, 1].plot(cumulative_time3, accuracy3, fmts[2], linewidth=2, markersize=4, markevery=20)
# axes[1, 1].set_xlabel('Total Time (s)')
# axes[1, 1].set_ylabel('Accuracy')
# axes[1, 1].set_title('Accuracy vs Total Time')
# axes[1, 1].grid(True, alpha=0.3)

# Create a legend
handles = [
    plt.Line2D([0], [0], color='red',  linestyle='-', linewidth=2, markersize=6, label=labels[0]),
    plt.Line2D([0], [0], color='green',  linestyle='-', linewidth=2, markersize=6, label=labels[1]),
    plt.Line2D([0], [0], color='blue',  linestyle='-', linewidth=2, markersize=6, label=labels[2])
]

ax.legend(handles, labels, loc='lower right')

plt.tight_layout()
plt.show()