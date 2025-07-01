import csv

import matplotlib.pyplot as plt
import numpy as np
from keras.api.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.api.layers import BatchNormalization, Dense, Dropout, Input
from keras.api.metrics import RootMeanSquaredError
from keras.api.models import Model
from keras.api.optimizers import Adam

# ラベルは、alphaとbetaの2つ
LABEL_COUNT = 2

EPOCHS = 250
BATCH_SIZE = 32


def load_data(filepath):
    """
    CSVファイルから特徴量(ヒストグラム)とラベル(alpha, beta)を読み込む関数
    """
    features = []
    labels = []
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            # ラベル（alpha, beta）と特徴量（hist）を分けてリストに追加
            labels.append([float(row[0]), float(row[1])])
            features.append(list(map(float, row[2:])))
    # NumPy配列に変換して返す
    return np.array(features), np.array(labels)


# 訓練データと検証データを読み込む
X_train, y_train = load_data("dataset/beta/train_histogram.csv")
X_val, y_val = load_data("dataset/beta/val_histogram.csv")

input_layer = Input(shape=(X_train.shape[1],), name="input_layer")
x = BatchNormalization()(input_layer)  # 入力層の直後に追加

x = Dense(256, activation="relu", name="hidden_layer_1")(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)  # ドロップアウト率を0.2〜0.5で調整

x = Dense(64, activation="relu", name="hidden_layer_2")(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(16, activation="relu", name="hidden_layer_3")(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(4, activation="relu", name="hidden_layer_4")(x)
output_layer = Dense(LABEL_COUNT, activation="linear", name="output_layer")(x)

# モデルを定義
model = Model(inputs=input_layer, outputs=output_layer, name="beta_model")

# モデルのコンパイル
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mean_squared_error",
    metrics=[RootMeanSquaredError(name="rmse")],
)

# 学習率の減衰を設定
reduce_lr = ReduceLROnPlateau(
    monitor="val_rmse", factor=0.5, patience=10, min_lr=1e-6, verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_rmse",  # 監視する指標
    patience=25,  # 25エポック改善が見られなければ停止
    restore_best_weights=True,  # 最も性能が良かった重みに戻す
    verbose=1,
)

# モデルの学習
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[reduce_lr],
    verbose=1,
)

# モデルの保存
model.save("models/data-analytics/beta_model_v2.keras")


# 学習曲線の保存
def save_learning_curve(history, filepath):
    """
    学習曲線を保存する関数
    """
    plt.figure(figsize=(12, 6))

    # 訓練データのRMSE
    plt.plot(history.history["rmse"], label="Train RMSE")

    # 検証データのRMSE
    plt.plot(history.history["val_rmse"], label="Validation RMSE")

    plt.title("Learning Curve")
    plt.xlabel("Epochs")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid()

    # グラフを保存
    plt.savefig(filepath)
    plt.close()


# 学習曲線を保存
save_learning_curve(history, "dataset/beta/learning_curve_v2.png")
