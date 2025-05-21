import keras
import numpy as np


def main():
    # 疑似データ
    # 64次元のデータを100個生成
    X = np.random.normal(0, 1, (100, 64))

    # 画像データを100個生成
    img_X = np.random.randint(0, 256, (100, 64, 64, 3), dtype="uint8")

    # 入力レイヤの定義
    inputs = keras.Input(shape=(64,))
    print(inputs.dtype)
    print(inputs.shape)

    img_inputs = keras.Input(shape=(64, 64, 3), dtype="uint8")
    print(img_inputs.dtype)
    print(img_inputs.shape)

    # モデルの定義
    model = keras.Model(inputs=inputs, outputs=inputs)
    model.compile(optimizer="adam", loss="mse")
    model.summary()

    path = "images/functional_model.png"
    keras.utils.plot_model(model, to_file=path, show_shapes=True)

    return 0


if __name__ == "__main__":
    main()
    main()
