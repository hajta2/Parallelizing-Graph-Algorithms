#!/usr/bin/python3
from mpl_toolkits.mplot3d import Axes3D
import fire
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

class plotter:
    title: str
    output: str

    def __init__(self, title: str, output: str = None):
        self.title = title
        self.output = output

    def plot(self, input_file):
        sns.set_theme()
        sns.set_context("paper", font_scale=1.75)
        runtimes = pd.read_csv(input_file)
        print(runtimes.columns)
        print(runtimes)

        # sns.relplot(data=runtimes, x="Density", y="CSR w/ MKL", kind="line", hue="Vertices", legend="full")
        # plt.title("CSR with MKL")
        # plt.xlabel("Sparsity")
        # plt.ylabel("Runtimes (microseconds)")
        # sns.relplot(data=runtimes, x="Density", y="CSR w/o MKL", kind="line", hue="Vertices", legend="full")
        # plt.title(self.title)
        # plt.xlabel("Sparsity")
        # plt.ylabel("Runtimes (microseconds)")
    
        sns.lineplot(data = runtimes, x="Density", y="Bandwidth", hue="Vertices", palette=['r','r','r','r','r','r','r','r','g'], legend=False)
        sns.lineplot(data = runtimes, x="Density", y="Const", hue="Vertices", palette=['b','b','b','b','b','b','b','b','b'], legend=False)
        plt.title("Bandwidth")
        plt.xlabel("Sparsity")
        plt.ylabel("Bandwidth (MB/s)")
        custom_lines = [Line2D([0], [0], color='r', lw=4),
                        Line2D([0], [0], color='b', lw=4),
                        Line2D([0], [0], color='g', lw=4)]
        plt.legend(custom_lines, ['VCL Row', 'Max Limit', 'MKL'], loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.autoscale()
        plt.savefig(self.title + ".pdf",bbox_inches="tight")
        plt.show()

if __name__ == "__main__":
    fire.Fire(plotter)
