# 作成した画像分類モデルを読み込む

import numpy as np
import tensorflow as tf
from keras.api.models import load_model
from keras.api.utils import img_to_array, load_img

model = load_model("models/cats_and_dogs_model.keras")  # 学習済みモデルの読み込み

# 画像のファイルパス（判別したい画像）
img_path = "images/neural-network/test-dog.jpeg"

# モデルに合わせて画像をリサイズ
img = load_img(img_path, target_size=(128, 128))  # 入力サイズにリサイズ
img_array = img_to_array(img)  # NumPy 配列に変換
img_array = tf.expand_dims(img_array, 0)  # バッチ次元を追加（1枚の画像でも必要）

# 予測
predictions = model(img_array)  # 出力はlogits（Softmax前のスコア）
score = tf.nn.softmax(predictions[0])  # Softmaxで確率に変換

# ラベル名（クラス名）は学習時の順に合わせること
class_names = ["cat", "dog"]  # train_ds.class_names と同じ順

# 結果出力
predicted_class = class_names[np.argmax(score)]
confidence = 100 * np.max(score)

print(f"この画像は「{predicted_class}」である可能性が {confidence:.2f}% です。")
