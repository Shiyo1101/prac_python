# 手書き文字認識のためのニューラルネットワーク

# 1. ライブラリのインポート
import numpy as np
from PIL import Image


def show_image(image):
    """
    画像を表示する関数
    :param image: 28x28の画像データ
    """
    img = Image.fromarray(np.uint8(image * 255))  # 0-1の値を0-255に変換
    img.show()


def cross_entropy(y_true, y_pred):
    """
    クロスエントロピー損失関数
    :param y_true: 正解ラベル
    :param y_pred: 予測ラベル
    :return: クロスエントロピー損失
    """
    # y_predの値が0または1になることを防ぐために、クリッピングを行う
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]


def softmax(x):
    """
    ソフトマックス関数
    :param x: 入力データ
    :return: ソフトマックス出力
    """
    exp_x = np.exp(x - np.max(x))  # オーバーフローを防ぐために最大値を引く
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)  # 各行ごとに正規化


def main():
    # 2. MNISTデータセットの読み込み
    mnist_data = np.load("datasets/mnist.npz")

    print(mnist_data.files)  # ['x_train', 'y_train', 'x_test', 'y_test']
    x_train = mnist_data["x_train"]  # トレーニングデータ
    y_train = mnist_data["y_train"]  # トレーニングラベル
    x_test = mnist_data["x_test"]  # テストデータ
    y_test = mnist_data["y_test"]  # テストラベル

    print(x_train.shape)  # (60000, 28, 28)
    print(y_train.shape)  # (60000,)
    print(x_test.shape)  # (10000, 28, 28)
    print(y_test.shape)  # (10000,)

    img = x_train[0]  # 最初の画像
    label = y_train[0]  # 最初のラベル
    print(img)  # 28x28の画像データ
    print(label)  # 5
    print(img.shape)  # (28, 28)

    # 3. 画像を表示
    show_image(img)  # 画像を表示

    # 4. 画像データを前処理
    x_train = (
        x_train.reshape(x_train.shape[0], -1) / 255.0
    )  # 28x28の画像を784次元に変換し、0-1に正規化
    x_test = x_test.reshape(x_test.shape[0], -1) / 255.0  # 同様にテストデータも前処理
    print(x_train.shape)  # (60000, 784)
    print(x_test.shape)  # (10000, 784)

    # 5. ラベルをone-hotエンコーディング
    num_classes = 10  # クラス数（0-9の数字）
    y_train_one_hot = np.zeros((y_train.shape[0], num_classes))  # (60000, 10)
    y_train_one_hot[np.arange(y_train.shape[0]), y_train] = (
        1  # 正解ラベルをone-hotエンコーディング
    )
    y_test_one_hot = np.zeros((y_test.shape[0], num_classes))  # (10000, 10)
    y_test_one_hot[np.arange(y_test.shape[0]), y_test] = (
        1  # 同様にテストデータもone-hotエンコーディング
    )
    print(y_train_one_hot.shape)  # (60000, 10)
    print(y_test_one_hot.shape)  # (10000, 10)

    # 6. ニューラルネットワークの初期化
    input_size = x_train.shape[1]  # 入力層のサイズ（784）
    hidden_size = 128  # 隠れ層のサイズ（128）
    output_size = num_classes  # 出力層のサイズ（10）
    learning_rate = 0.01  # 学習率
    epochs = 1000  # エポック数
    batch_size = 32  # バッチサイズ
    num_batches = x_train.shape[0] // batch_size  # バッチ数
    weights_input_hidden = (
        np.random.randn(input_size, hidden_size) * 0.01
    )  # 入力層 -> 隠れ層の重み
    weights_hidden_output = (
        np.random.randn(hidden_size, output_size) * 0.01
    )  # 隠れ層 -> 出力層の重み
    biases_hidden = np.zeros((1, hidden_size))  # 隠れ層のバイアス
    biases_output = np.zeros((1, output_size))  # 出力層のバイアス

    # 7. 学習ループ
    for epoch in range(epochs):
        for batch in range(num_batches):
            # バッチデータの取得
            x_batch = x_train[batch * batch_size : (batch + 1) * batch_size]
            y_batch = y_train_one_hot[batch * batch_size : (batch + 1) * batch_size]

            # 順伝播
            hidden_input = np.dot(x_batch, weights_input_hidden) + biases_hidden
            hidden_output = np.tanh(hidden_input)  # 隠れ層の活性化関数
            final_input = np.dot(hidden_output, weights_hidden_output) + biases_output
            final_output = softmax(final_input)  # 出力層の活性化関数

            # 誤差の計算
            loss = cross_entropy(y_batch, final_output)

            # 誤差逆伝播
            output_error = final_output - y_batch  # 出力層の誤差
            hidden_error = np.dot(output_error, weights_hidden_output.T)  # 隠れ層の誤差

            # 重みとバイアスの更新
            weights_hidden_output -= (
                np.dot(hidden_output.T, output_error) / batch_size
            )  # 隠れ層 -> 出力層の重み更新
            biases_output -= (
                np.sum(output_error, axis=0, keepdims=True) / batch_size
            )  # 出力層のバイアス更新
            weights_input_hidden -= (
                np.dot(x_batch.T, hidden_error * (1 - hidden_output**2)) / batch_size
            )  # 入力層 -> 隠れ層の重み更新
            biases_hidden -= (
                np.sum(hidden_error * (1 - hidden_output**2), axis=0, keepdims=True)
                / batch_size
            )  # 隠れ層のバイアス更新

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

    # 8. テストデータでの評価
    test_hidden_input = np.dot(x_test, weights_input_hidden) + biases_hidden
    test_hidden_output = np.tanh(test_hidden_input)
    test_final_input = np.dot(test_hidden_output, weights_hidden_output) + biases_output
    test_final_output = softmax(test_final_input)
    test_loss = cross_entropy(y_test_one_hot, test_final_output)
    test_accuracy = np.mean(
        np.argmax(test_final_output, axis=1) == np.argmax(y_test_one_hot, axis=1)
    )
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    # 9. テストデータの予測
    predictions = np.argmax(test_final_output, axis=1)  # 予測ラベル
    print(predictions)  # 予測ラベルを表示

    # 10. 予測結果を表示
    for i in range(5):  # 最初の5つの予測結果を表示
        print(f"Predicted: {predictions[i]}, Actual: {y_test[i]}")
        show_image(x_test[i].reshape(28, 28))  # 画像を表示

    # 11. 予測結果を保存
    np.savez("predictions.npz", predictions=predictions)  # 予測結果を保存
    print("Predictions saved to predictions.npz")  # 保存完了メッセージ

    # 12. 予測結果を読み込み
    loaded_data = np.load("predictions.npz")  # 保存した予測結果を読み込み
    loaded_predictions = loaded_data["predictions"]  # 読み込んだ予測結果
    print(loaded_predictions)  # 読み込んだ予測結果を表示
    slice

    # 13. 予測結果を表示
    for i in range(5):  # 最初の5つの予測結果を表示
        print(f"Predicted: {loaded_predictions[i]}, Actual: {y_test[i]}")
        show_image(x_test[i].reshape(28, 28))  # 画像を表示


if __name__ == "__main__":
    main()
