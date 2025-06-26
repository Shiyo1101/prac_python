import numpy as np
from scipy import stats

# 生成する乱数の個数（1M）
N = 1000000

# 平均と標準偏差の範囲設定
MU_RANGE = (-5, 5)
SIGMA_RANGE = (0.1, 2)

# 生成するヒストグラムの個数
HISTOGRAM_COUNT = 1500

# ヒストグラムのBINS数
HISTOGRAM_BINS = 1000

for i in range(HISTOGRAM_COUNT):
    # パラメータをランダムに設定
    mu = np.random.uniform(*MU_RANGE)
    sigma = np.random.uniform(*SIGMA_RANGE)

# 正規分布に従う乱数を生成
data = stats.norm.rvs(loc=mu, scale=sigma, size=N)

# ヒストグラムを計算
hist, bin_edges = np.histogram(data, bins=HISTOGRAM_BINS, range=(-10, 10), density=True)

# hist, mu, sigmaをcsv形式で追加書き込み
with open("dataset/gauss/histogram.csv", "a") as f:
    f.write(f"{mu},{sigma}," + ",".join(map(str, hist)) + "\n")
