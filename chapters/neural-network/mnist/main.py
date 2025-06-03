import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.api import layers, models
from keras.api.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix


def main():
    # データセットの読み込み
    train_ds = np.load("dataset/mnist/mnist_train.npz")
    val_ds = np.load("dataset/mnist/mnist_val.npz")
    train_images = train_ds["images"]
    train_labels = train_ds["labels"]
    val_images = val_ds["images"]
    val_labels = val_ds["labels"]

    # データセットの前処理
    train_images = train_images.astype("float32") / 255.0
    val_images = val_images.astype("float32") / 255.0
    train_labels = train_labels.astype("int32")
    val_labels = val_labels.astype("int32")

    # (N, 1, 28, 28) → (N, 28, 28)
    if train_images.ndim == 4 and train_images.shape[1] == 1:
        train_images = np.squeeze(train_images, axis=1)
        val_images = np.squeeze(val_images, axis=1)

    # (N, 28, 28) → (N, 28, 28, 1)
    train_images = np.expand_dims(train_images, -1)
    val_images = np.expand_dims(val_images, -1)

    print("train_images.shape", train_images.shape)
    print("train_labels.shape", train_labels.shape)
    print("val_images.shape", val_images.shape)
    print("val_labels.shape", val_labels.shape)

    # モデルの定義
    model = models.Sequential()
    model.add(layers.Input(shape=(28, 28, 1)))

    # Conv Block 1
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    # Conv Block 2
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    # Conv Block 3
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))

    # Dense Block
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))

    # モデルの概要
    model.summary()

    # モデルのコンパイル
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # コールバック関数の設定
    lr_reducer = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1
    )

    # 早期終了のコールバック
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    )

    # モデルの訓練
    history = model.fit(
        train_images,
        train_labels,
        epochs=100,
        batch_size=64,
        validation_data=(val_images, val_labels),
        callbacks=[lr_reducer, early_stopping],  # コールバック追加
    )

    # 学習曲線のプロットと保存（train, valのAccuracy Curve）
    plt.figure(figsize=(7, 5))
    epochs = range(1, len(history.history["accuracy"]) + 1)
    plt.plot(epochs, history.history["accuracy"], label="train_acc")
    plt.plot(epochs, history.history["val_accuracy"], label="val_acc")

    # 10エポックごとに val_acc の値を点で表示
    for i in range(9, len(epochs), 10):
        plt.scatter(epochs[i], history.history["val_accuracy"][i], color="red")
        plt.text(
            epochs[i],
            history.history["val_accuracy"][i],
            f"{history.history['val_accuracy'][i]:.3f}",
            fontsize=8,
            ha="center",
            va="bottom",
            color="red",
        )

    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("models/mnist_accuracy_curve.png")
    plt.close()

    # モデルの保存
    model.save("models/mnist_model.keras")

    # モデルの評価
    loss, accuracy = model.evaluate(val_images, val_labels)
    print(f"Validation Accuracy: {accuracy:.2f}")
    print(f"Validation Loss: {loss:.2f}")

    # モデルの予測
    val_pred = model.predict(val_images)
    val_pred_classes = np.argmax(val_pred, axis=1)

    # 混同行列の作成
    cm = confusion_matrix(val_labels, val_pred_classes)

    # 0～9すべての正解ラベルごとに、予測分布の表を作成・保存
    df_count = pd.DataFrame(cm, columns=[f"pred_{i}" for i in range(10)])
    df_count.index = [f"true_{i}" for i in range(10)]
    df_count.to_csv("models/mnist_confusion_count.csv")

    # 割合の表も作成・保存（各行で正規化）
    df_ratio = df_count.div(df_count.sum(axis=1), axis=0)
    df_ratio.to_csv("models/mnist_confusion_ratio.csv")


if __name__ == "__main__":
    main()
