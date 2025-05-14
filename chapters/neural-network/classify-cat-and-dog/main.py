# ライブラリのインポート
import cv2
import tensorflow as tf
from keras import layers, models

# データセットのインポート
from load_dataset import class_names, train_ds, val_ds


def main():
    num_classes = 2  # 猫と犬の2クラス

    # 1. モデルの定義
    model = models.Sequential()
    model.add(layers.Input(shape=(128, 128, 3)))
    model.add(layers.Conv2D(16, 3, padding="same", activation="relu"))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(32, 3, padding="same", activation="relu"))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(64, 3, padding="same", activation="relu"))
    model.add(layers.MaxPooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(num_classes))

    # 2. モデルのコンパイル
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # 3. モデルの訓練
    model.fit(train_ds, validation_data=val_ds, epochs=10)

    # 4. モデルの保存
    model.save("models/cats_and_dogs_model.keras")

    # 5. モデルの評価
    loss, accuracy = model.evaluate(val_ds)
    print(f"Validation Accuracy: {accuracy:.2f}")
    print(f"Validation Loss: {loss:.2f}")

    # 6. モデルの予測
    # 予測のための画像を読み込む
    image = cv2.imread("images/test-cat.jpg")
    image = cv2.resize(image, (128, 128))
    image = image / 255.0  # ピクセルを0〜1に正規化
    image = image.reshape(1, 128, 128, 3)  # バッチサイズを追加
    predictions = model.predict(image)
    predicted_class = tf.argmax(predictions[0]).numpy()
    print(f"Predicted class: {class_names[predicted_class]}")

    # 7. モデルの可視化
    # モデルの構造を可視化
    model.summary()

    # モデルのアーキテクチャを画像として保存
    tf.keras.utils.plot_model(
        model,
        to_file="models/cats_and_dogs_model.png",
        show_shapes=True,
        show_layer_names=True,
    )


if __name__ == "__main__":
    main()
