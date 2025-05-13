# tensorflow, kerasを使用した簡単なCNNの構築

import matplotlib.pyplot as plt
from keras import layers, models
from keras.src.datasets import mnist


def main():
    # MNISTデータセットの読み込み
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # データの前処理
    x_train = x_train.reshape((60000, 28, 28, 1))
    x_test = x_test.reshape((10000, 28, 28, 1))
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")

    # モデルの構築
    # Sequentialモデルを使用してCNNを構築
    model = models.Sequential()
    model.add(layers.Input(shape=(28, 28, 1)))  # Inputレイヤーを追加
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))  # 畳み込み層を追加
    model.add(layers.MaxPooling2D((2, 2)))  # プーリング層を追加
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))  # 畳み込み層を追加
    model.add(layers.MaxPooling2D((2, 2)))  # プーリング層を追加
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))  # 畳み込み層を追加
    model.add(layers.Flatten())  # Flatten層を追加
    model.add(layers.Dense(64, activation="relu"))  # 全結合層を追加
    model.add(layers.Dense(10, activation="softmax"))  # 出力層を追加
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    # モデルの学習
    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

    # モデルの評価
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print("test_loss:", test_loss)
    print("test_acc:", test_acc)

    # モデルの保存
    model.save("models/mnist_cnn_model.keras")

    # モデルの読み込み
    loaded_model = models.load_model("models/mnist_cnn_model.keras")

    # モデルの再評価
    test_loss, test_acc = loaded_model.evaluate(x_test, y_test)
    print("test_loss:", test_loss)
    print("test_acc:", test_acc)

    # モデルの予測
    predictions = loaded_model.predict(x_test)
    print("predictions.shape:", predictions.shape)
    print("predictions[0]:", predictions[0])
    print("predictions[0].shape:", predictions[0].shape)
    print("predictions[0].argmax():", predictions[0].argmax())
    print("predictions[0].argmax()の値:", predictions[0].argmax())
    print("y_test[0]:", y_test[0])
    print("y_test[0].shape:", y_test[0].shape)

    # 予測結果の表示
    plt.imshow(x_test[0].reshape(28, 28), cmap="gray")
    plt.title(f"Predicted: {predictions[0].argmax()}, Actual: {y_test[0]}")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
