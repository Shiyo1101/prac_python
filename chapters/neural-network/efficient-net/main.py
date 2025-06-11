# EfficientNet Test

import numpy as np
from keras.api.applications import EfficientNetV2L
from keras.api.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.api.preprocessing import image


def main():
    model = EfficientNetV2L(weights="imagenet")
    print("\n◆Model:")
    print(f"{model.name}")

    img_path = "images/cat.jpg"
    img = image.load_img(img_path, target_size=(480, 480))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # モデルを使って予測
    print("\n◆Predict:")
    preds = model.predict(x)
    preds_top = decode_predictions(preds, top=3)[0]

    # 結果を表示
    print(f"{preds_top[0][1]} {preds_top[0][2] * 100:.2f}%")


if __name__ == "__main__":
    main()
