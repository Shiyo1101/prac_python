# ベータ分布に従うデータのヒストグラムを作成する

import os

import numpy as np
from scipy import stats

# 生成する乱数の個数（1M）
N = 1000000

# 生成するヒストグラムの個数
HISTOGRAM_COUNT = 1500

# alphaとbetaの範囲設定
ALPHA_RANGE = (0.8, 10.0)
BETA_RANGE = (0.8, 10.0)

# ヒストグラムのBINS数
HISTOGRAM_BINS = 1000

# 出力ディレクトリの作成
output_dir = "dataset/beta"
os.makedirs(output_dir, exist_ok=True)

# 出力ファイルパス
output_file = os.path.join(output_dir, "histogram.csv")

# ヒストグラムデータを生成してCSVに書き込み
for i in range(HISTOGRAM_COUNT):
    # alphaとbetaをランダムに設定（各1つ）
    alpha = np.random.uniform(*ALPHA_RANGE)
    beta = np.random.uniform(*BETA_RANGE)
    print(f"Generating histogram {i + 1}/{HISTOGRAM_COUNT}: alpha={alpha}, beta={beta}")

    # ベータ分布に従う乱数を生成
    data = stats.beta.rvs(alpha, beta, size=N)

    # ヒストグラムを計算
    hist, bin_edges = np.histogram(
        data, bins=HISTOGRAM_BINS, range=(0, 1), density=True
    )

    # hist, alpha, betaをcsv形式で追加書き込み
    with open(output_file, "a") as f:
        f.write(f"{alpha},{beta}," + ",".join(map(str, hist)) + "\n")
