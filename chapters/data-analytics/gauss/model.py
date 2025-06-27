import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from keras.api.callbacks import ReduceLROnPlateau
from keras.api.layers import Dense, Input
from keras.api.metrics import RootMeanSquaredError
from keras.api.models import Model
from keras.api.optimizers import Adam

# ラベルは、muとsigmaの2つ
LABEL_COUNT = 2

EPOCHS = 250
BATCH_SIZE = 32


def load_data(filepath):
    """
    CSVファイルから特徴量(ヒストグラム)とラベル(mu, sigma)を読み込む関数
    """
    features = []
    labels = []
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            # ラベル（mu, sigma）と特徴量（hist）を分けてリストに追加
            labels.append([float(row[0]), float(row[1])])
            features.append(list(map(float, row[2:])))
    # NumPy配列に変換して返す
    return np.array(features), np.array(labels)


# 訓練データと検証データを読み込む
X_train, y_train = load_data("dataset/gauss/train_histogram.csv")
X_val, y_val = load_data("dataset/gauss/val_histogram.csv")

input_layer = Input(shape=(X_train.shape[1],), name="input_layer")

x = Dense(256, activation="relu", name="hidden_layer_1")(input_layer)
x = Dense(64, activation="relu", name="hidden_layer_2")(x)
x = Dense(16, activation="relu", name="hidden_layer_3")(x)
x = Dense(4, activation="relu", name="hidden_layer_4")(x)
output_layer = Dense(LABEL_COUNT, activation="linear", name="output_layer")(x)

# モデルを定義
model = Model(inputs=input_layer, outputs=output_layer, name="gauss_model")

# モデルをコンパイル
# 損失関数には一般的に使われる平均二乗誤差(mean squared error)を使用
# 評価指標(metrics)としてRMSEを監視する
model.compile(
    optimizer=Adam(),
    loss="mean_squared_error",
    metrics=[RootMeanSquaredError(name="rmse")],
)

# モデルの概要を表示
model.summary()

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",  # 監視対象
    factor=0.5,  # 学習率を 1/2 に
    patience=20,  # 10エポック改善が見られなければ実行
    min_lr=1e-6,  # 下限の学習率
    verbose=1,
)

# モデルを学習
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    callbacks=[reduce_lr],
)

# モデルの保存先ディレクトリが存在しない場合に作成
os.makedirs("models", exist_ok=True)

# モデルを保存
model.save("models/gauss_model.keras")

print("\nモデルの学習と保存が完了しました。")


# モデルの予測をグラフでプロットする
def plot_predictions(model, X, y_true):
    """
    モデルの予測をグラフでプロットする関数
    """
    # 予測値を計算
    y_pred = model.predict(X)

    # muとsigmaの予測値を取得
    mu_pred = y_pred[:, 0]
    sigma_pred = y_pred[:, 1]
    mu_true = y_true[:, 0]
    sigma_true = y_true[:, 1]

    # グラフの設定
    plt.figure(figsize=(12, 6))

    # muの予測と実際の値をプロット
    plt.subplot(1, 2, 1)
    plt.scatter(mu_true, mu_pred, alpha=0.5)
    plt.plot(
        [min(mu_true), max(mu_true)],
        [min(mu_true), max(mu_true)],
        color="red",
        linestyle="--",
    )
    plt.title("Predicted vs True mu")
    plt.xlabel("True mu")
    plt.ylabel("Predicted mu")
    plt.grid()

    # sigmaの予測と実際の値をプロット
    plt.subplot(1, 2, 2)
    plt.scatter(sigma_true, sigma_pred, alpha=0.5)
    plt.plot(
        [min(sigma_true), max(sigma_true)],
        [min(sigma_true), max(sigma_true)],
        color="red",
        linestyle="--",
    )
    plt.title("Predicted vs True sigma")
    plt.xlabel("True sigma")
    plt.ylabel("Predicted sigma")
    plt.grid()

    # グラフを保存
    plt.savefig("images/gauss/predictions_plot.png")


# 訓練データに対する予測をプロット
plot_predictions(model, X_train, y_train)
