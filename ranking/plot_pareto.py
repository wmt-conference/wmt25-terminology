
import matplotlib.pyplot as plt


# Flag: label only Pareto frontier points or all points
LABEL_ONLY_FRONTIER = False
# Flag: whether to plot the Pareto frontier using red dotted lines.
PLOT_PARETO_FRONTIER = True

# Adjust the plot size.
px = 1/plt.rcParams['figure.dpi']  # pixel in inches
plt.subplots(figsize=(800*px, 800*px))

# Example score tuples.
# (term_acc, quality (chrF2++), label)
scores = [
(0.99, 50, "team 1"),
(0.98, 70, "team 2"),
(0.95, 75, "team 3"),
(0.88, 67, "team 4"),
(0.88, 65, "team 5"),
(0.80, 60, "team 6"),
(0.78, 61, "team 7"),
(0.72, 61, "team 8"),
]

# Determine Pareto-optimal points
pareto_front = []
for i, (x, y, label) in enumerate(scores):
    dominated = False
    for j, (x2, y2, _) in enumerate(scores):
        if j != i and x2 >= x and y2 >= y and (x2 > x or y2 > y):
            dominated = True
            break
    if not dominated:
        pareto_front.append((x, y, label))

# Sort the frontier for plotting (left to right)
pareto_front.sort(key=lambda p: p[0])

# Unpack coordinates
x_all, y_all, labels_all = zip(*scores)
x_pareto, y_pareto, labels_pareto = zip(*pareto_front)

# Construct inward staircase: vertical then horizontal
x_stair = [x_pareto[0]]
y_stair = [y_pareto[0]]
for i in range(1, len(pareto_front)):
    x_stair.append(x_pareto[i - 1])
    y_stair.append(y_pareto[i])
    x_stair.append(x_pareto[i])
    y_stair.append(y_pareto[i])

# Plot all points
plt.scatter(x_all, y_all, color='blue', label='Systems', marker='x', s=30)

if PLOT_PARETO_FRONTIER:
    # Plot Pareto frontier as inward dotted staircase
    plt.plot(x_stair, y_stair, linestyle=':', color='red', linewidth=3, label='Pareto Frontier')

# Label points
if LABEL_ONLY_FRONTIER:
    for x, y, label in pareto_front:
        plt.text(x-0.01, y-0.01, label, fontsize=10, ha='right', va='top')
else:
    for x, y, label in scores:
        plt.text(x-0.01, y-0.01, label, fontsize=10, ha='right', va='top')

# Labels and legend
plt.xlabel('Terminology Accuracy')
plt.ylabel('COMET22')
plt.title('Pareto Optimal')
plt.legend()
plt.grid(True)

plt.savefig('pareto.png')
