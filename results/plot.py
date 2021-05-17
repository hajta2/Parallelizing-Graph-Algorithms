#!/usr/bin/python3
from mpl_toolkits.mplot3d import Axes3D
import fire
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

class plotter:
    title: str
    output: str

    def __init__(self, title: str, output: str = None):
        self.title = title
        self.output = output

    def plot(self, input_file):
        sns.set_theme()
        runtimes = pd.read_csv(input_file)
        print(runtimes.columns)
        print(runtimes)
        #sns.relplot(data=runtimes, x="Density", y="CSR w/ MKL", kind="line", hue="Vertices")
        # plt.title("CSR w/ MKL")
        # plt.xlabel("Sparsity")
        # plt.ylabel("Runtimes (microseconds)")
        # sns.relplot(data=runtimes, x="Density", y="CSR w/o MKL", kind="line", hue="Vertices")
        # plt.title("CSR w/o MKL")
        # plt.show()
        sns.lineplot(data = runtimes, x="Density", y="CSR w/ MKL", hue="Vertices", palette=['red'], legend=False)
        sns.lineplot(data = runtimes, x="Density", y="CSR w/o MKL", hue="Vertices", palette=['blue'], legend=False)
        # # sns.lineplot(data = runtimes, x="Density", y="CSR w/ MKL", hue="Vertices")
        # # sns.lineplot(data = runtimes, x="Density", y="CSR w/o MKL", hue="Vertices")
        plt.xlabel("Sparsity")
        plt.ylabel("Runtimes (microseconds)")
        plt.show()
        plt.savefig(self.title + ".png")

if __name__ == "__main__":
    fire.Fire(plotter)
