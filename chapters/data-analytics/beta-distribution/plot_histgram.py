import csv

import matplotlib.pyplot as plt
import numpy as np


def main():
    # alpha, beta, histをcsvから読み込み
    data = []
    with open("dataset/beta/histogram.csv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:  # 空行をスキップ
                continue
            alpha = float(row[0])
            beta = float(row[1])
            hist = list(map(float, row[2:]))
            data.append((alpha, beta, hist))

    # １行分ヒストグラムをプロット
    plt.figure(figsize=(12, 6))
    for alpha, beta, hist in data:
        # ヒストグラムのx軸の値を計算
        bin_edges = np.linspace(0, 1, len(hist) + 1)
        x = (bin_edges[:-1] + bin_edges[1:]) / 2

        # ヒストグラムをプロット
        plt.plot(x, hist, label=f"alpha={alpha:.2f}, beta={beta:.2f}")

    plt.title("Histograms of Beta Distributions")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.grid()
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 5)
    plt.show()

    return


if __name__ == "__main__":
    main()
