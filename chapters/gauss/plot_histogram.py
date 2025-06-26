import csv

import matplotlib.pyplot as plt
import numpy as np


def main():
    # mu, sigma, histをcsvから読み込み
    data = []
    with open("dataset/gauss/histogram.csv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            mu = float(row[0])
            sigma = float(row[1])
            hist = list(map(float, row[2:]))
            data.append((mu, sigma, hist))

    # １行分ヒストグラムをプロット
    plt.figure(figsize=(12, 6))
    for mu, sigma, hist in data:
        # ヒストグラムのx軸の値を計算
        bin_edges = np.linspace(-10, 10, len(hist) + 1)
        x = (bin_edges[:-1] + bin_edges[1:]) / 2

        # ヒストグラムをプロット
        plt.plot(x, hist, label=f"mu={mu:.2f}, sigma={sigma:.2f}")

    plt.title("Histograms of Normal Distributions")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.grid()
    plt.legend()
    plt.xlim(-10, 10)
    plt.ylim(0, 0.5)
    plt.show()

    return


if __name__ == "__main__":
    main()
