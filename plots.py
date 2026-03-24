import matplotlib.pyplot as plt
import numpy as np

def plot_class_distribution(data, title, show = True):
    counts = np.unique_counts(data)
    plt.bar(counts.values, counts.counts)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title(title)
    plt.xticks(counts.values)
    if show:
        plt.show()