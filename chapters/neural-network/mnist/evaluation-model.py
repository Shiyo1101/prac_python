# 作成したモデルの評価

import matplotlib.pyplot as plt
import numpy as np
from keras import models, utils


def main():
    # テスト用のデータセットの読み込み
    test_ds = np.load("dataset/mnist/mnist_test.npz")
    test_images = test_ds["images"]
    test_labels = test_ds["labels"]

    # データセットの前処理
    test_images = test_images.astype("float32") / 255.0
    test_labels = test_labels.astype("int32")

    # (N, 1, 28, 28) → (N, 28, 28)
    if test_images.ndim == 4 and test_images.shape[1] == 1:
        test_images = np.squeeze(test_images, axis=1)

    # (N, 28, 28) → (N, 28, 28, 1)
    test_images = np.expand_dims(test_images, -1)

    print("test_images.shape", test_images.shape)
    print("test_labels.shape", test_labels.shape)

    # モデルの読み込み
    model = models.load_model("models/mnist_model.keras")

    # モデルの評価
    loss, accuracy = model.evaluate(test_images, test_labels)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Loss: {loss:.4f}")

    # モデルの予測
    predictions = model.predict(test_images)
    predicted_classes = np.argmax(predictions, axis=1)
    print(f"Predicted classes: {predicted_classes}")
    print(f"True classes: {test_labels}")

    # 予測結果の表示
    for i in range(10):
        print(
            f"Image {i}: Predicted class: {predicted_classes[i]}, True class: {test_labels[i]}"
        )

    # 予測結果の可視化
    plt.figure(figsize=(10, 10))
    for i in range(10):
        plt.subplot(5, 5, i + 1)
        plt.imshow(test_images[i].reshape(28, 28), cmap="gray")
        plt.title(f"Pred: {predicted_classes[i]}\nTrue: {test_labels[i]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

    # モデルのアーキテクチャを画像として保存
    utils.plot_model(
        model,
        to_file="models/mnist_model_arc.png",
        show_shapes=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=True,
        dpi=96,
    )


if __name__ == "__main__":
    main()
