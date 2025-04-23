# XOR問題を解くためのシンプルなニューラルネットワーク

import numpy as np

from common.functions import sigmoid


# シグモイド関数の微分
def sigmoid_derivative(x):
    return x * (1 - x)


# トレーニングデータ（XOR問題）
# 入力データ（4つのサンプル、2つの特徴）
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# 出力データ（XORの結果）
# 0 XOR 0 = 0
# 0 XOR 1 = 1
# 1 XOR 0 = 1
# 1 XOR 1 = 0
y = np.array([[0], [1], [1], [0]])

# 重みの初期化（ランダムな小さい値）
# 入力層 -> 隠れ層（2入力 -> 2ユニット）
weights_input_hidden = np.random.uniform(-1, 1, (2, 2))

# 隠れ層 -> 出力層（2ユニット -> 1出力）
weights_hidden_output = np.random.uniform(-1, 1, (2, 1))

# 学習率（重みの更新のステップサイズ）
learning_rate = 0.1

# エポック数（学習の繰り返し回数）
epochs = 10000

# 学習ループ
for epoch in range(epochs):
    # --- 順伝播（Forward Propagation） ---
    # 入力から隠れ層へ
    hidden_input = np.dot(X, weights_input_hidden)
    hidden_output = sigmoid(hidden_input)

    # 隠れ層から出力層へ
    final_input = np.dot(hidden_output, weights_hidden_output)
    final_output = sigmoid(final_input)

    # --- 誤差の計算（出力と正解の差）---
    error = y - final_output

    # --- 誤差逆伝播（Backpropagation） ---
    # 出力層のデルタ（微分 × 誤差）
    output_delta = error * sigmoid_derivative(final_output)

    # 隠れ層のデルタ（出力層のデルタを重みによって逆伝播）
    hidden_error = output_delta.dot(weights_hidden_output.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_output)

    # --- 重みの更新 ---
    # 隠れ層 -> 出力層の重み更新
    weights_hidden_output += hidden_output.T.dot(output_delta) * learning_rate

    # 入力層 -> 隠れ層の重み更新
    weights_input_hidden += X.T.dot(hidden_delta) * learning_rate

    # --- 途中で誤差を表示（1000エポックごと） ---
    if epoch % 1000 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# --- 学習結果の出力 ---
print("\n最終出力（学習後）:")
print(final_output)
print(final_output)

# --- テストデータの出力 ---
print("\nテストデータ（XORの結果）:")
for i in range(len(X)):
    print(f"入力: {X[i]}, 出力: {final_output[i]}")
