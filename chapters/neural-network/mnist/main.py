import numpy as np
from keras.api import layers, models
from keras.api.callbacks import ReduceLROnPlateau


def main():
    # データセットの読み込み
    train_ds = np.load("dataset/mnist/mnist_train.npz")
    val_ds = np.load("dataset/mnist/mnist_test.npz")
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
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))  # ドロップアウト層
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

    # モデルの訓練
    model.fit(
        train_images,
        train_labels,
        epochs=100,
        batch_size=64,
        validation_data=(val_images, val_labels),
        callbacks=[lr_reducer],  # コールバック追加
    )

    # モデルの保存
    model.save("models/mnist_model.keras")

    # モデルの評価
    loss, accuracy = model.evaluate(val_images, val_labels)
    print(f"Validation Accuracy: {accuracy:.2f}")
    print(f"Validation Loss: {loss:.2f}")


if __name__ == "__main__":
    main()
