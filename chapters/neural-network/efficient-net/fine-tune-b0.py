import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from keras import Model, Sequential, losses
from keras.api.applications.efficientnet_v2 import EfficientNetV2B0, preprocess_input
from keras.api.callbacks import ReduceLROnPlateau
from keras.api.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    Lambda,
    RandomFlip,
    RandomRotation,
    RandomZoom,
    RandomTranslation,
)
from keras.api.optimizers import AdamW


# ==============================================================================
# ステップ0: MirroredStrategy のセットアップ
# ==============================================================================
print("ステップ0: MirroredStrategy のセットアップを開始します...")
# 検出されたGPUデバイスのリストを取得
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        # すべてのGPUに対してメモリ増加を許可
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        strategy = tf.distribute.MirroredStrategy()
        print(f"検出されたデバイス数: {strategy.num_replicas_in_sync} (GPU)")
    except RuntimeError as e:
        # メモリ増加の設定中にエラーが発生した場合
        print(e)
        # エラーが発生した場合はデフォルト戦略を使用
        strategy = tf.distribute.get_strategy()
        print("MirroredStrategy のセットアップに失敗しました。デフォルトの戦略を使用します。")
else:
    # GPUが検出されない場合はデフォルト戦略 (CPU) を使用
    strategy = tf.distribute.get_strategy()
    print("GPUが検出されませんでした。デフォルトの戦略 (CPU) を使用します。")


# ---
# # ステップ1: データセットの準備 (Food101)
# ---
print("ステップ1: データセットの準備を開始します...")
(train_ds, validation_ds), ds_info = tfds.load(
    "food101",
    split=["train", "validation"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)
NUM_CLASSES = ds_info.features["label"].num_classes
print(f"データセットのロード完了。クラス数: {NUM_CLASSES}")


# ---
# # ステップ2: データの前処理とデータパイプラインの構築 (MixUp対応)
# ---
print("\nステップ2: データの前処理パイプラインを構築します...")
IMG_SIZE = 224  # EfficientNetV2B0の入力サイズ
BATCH_SIZE = 32


def preprocess_img(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label


def to_one_hot(image, label):
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label


AUTOTUNE = tf.data.AUTOTUNE

# 先にシャッフルとバッチ化を行う
train_ds_one = (
    train_ds.map(preprocess_img, num_parallel_calls=AUTOTUNE)
    .map(to_one_hot, num_parallel_calls=AUTOTUNE)
    .shuffle(buffer_size=1000)
    .batch(BATCH_SIZE)
)
train_ds_two = (
    train_ds.map(preprocess_img, num_parallel_calls=AUTOTUNE)
    .map(to_one_hot, num_parallel_calls=AUTOTUNE)
    .shuffle(buffer_size=1000)
    .batch(BATCH_SIZE)
)


def mixup(ds_one, ds_two, alpha=0.2):
    ds = tf.data.Dataset.zip((ds_one, ds_two))

    def _mixup(batch_one, batch_two):
        images1, labels1 = batch_one
        images2, labels2 = batch_two
        batch_size = tf.shape(images1)[0]

        # 1次元のラムダ (混ぜ合わせる比率) を生成
        lambda_val = tf.random.uniform(shape=(batch_size,), minval=0.0, maxval=1.0)

        # 画像を混ぜるため、ラムダを (バッチサイズ, 1, 1, 1) の形状に変形
        lambda_images = tf.reshape(lambda_val, (batch_size, 1, 1, 1))

        # ラベルを混ぜるため、ラムダを (バッチサイズ, 1) の形状に変形
        lambda_labels = tf.reshape(lambda_val, (batch_size, 1))

        # それぞれ適切な形状のラムダを使ってミックス
        mixed_images = (images1 * lambda_images) + (images2 * (1 - lambda_images))
        mixed_labels = (labels1 * lambda_labels) + (labels2 * (1 - lambda_labels))

        return mixed_images, mixed_labels

    return ds.map(_mixup, num_parallel_calls=AUTOTUNE)


# バッチ化済みのデータセットにMixUpを適用し、最後にprefetchする
train_pipeline = mixup(train_ds_one, train_ds_two).prefetch(buffer_size=AUTOTUNE)

# 検証パイプライン
validation_pipeline = (
    validation_ds.map(preprocess_img, num_parallel_calls=AUTOTUNE)
    .map(to_one_hot, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(buffer_size=AUTOTUNE)
)
print("データパイプライン構築完了。")


# ---
# # ステップ3: データ拡張を含むモデル構築
# ---
print("\nステップ3: データ拡張を含むモデルを構築します...")

# ここから分散戦略のスコープ内
with strategy.scope():
    # データ拡張層を定義
    data_augmentation = Sequential(
        [
            RandomFlip("horizontal"),  # 水平方向のランダムフリップ
            RandomRotation(0.2),  # 回転
            RandomZoom(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2)),  # ズーム
            RandomTranslation(height_factor=0.1, width_factor=0.1),  # 平行移動
        ],
        name="data_augmentation",
    )

    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = data_augmentation(inputs)  # まずデータ拡張を適用
    x = Lambda(lambda img: preprocess_input(tf.cast(img, tf.float32)))(
        x
    )  # 次にEfficientNet用の前処理を適用

    base_model = EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
    )

    # base_modelの層の総数を確認
    print(f"EfficientNetV2BL の層の総数: {len(base_model.layers)}")

    base_model.trainable = False  # まずはベースモデルを凍結

    x = base_model.output
    x = GlobalAveragePooling2D(name="avg_pool")(x)
    x = Dense(512, activation="relu", name="dense_intermediate")(x)  # 中間層
    x = BatchNormalization(name="batch_norm")(x)  # バッチ正規化
    x = Dropout(0.4, name="dropout_top")(x)  # ドロップアウト強化
    outputs = Dense(NUM_CLASSES, activation="softmax", name="dense_output")(x)

    # モデルの構築
    model = Model(inputs, outputs, name="food101_efficientnetv2b0")

    # ---
    # # ステップ4: 特徴抽出のための初期学習 (スコープ内へ移動)
    # ---
    print("\nステップ4: 特徴抽出のための初期学習を開始します...")
    initial_epochs = 10
    model.compile(
        optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-5),  # AdamWオプティマイザを使用
        loss=losses.CategoricalCrossentropy(label_smoothing=0.1),  # ラベルスムージングを適用
        metrics=["accuracy"],
    )

    history = model.fit(
        train_pipeline, epochs=initial_epochs, validation_data=validation_pipeline
    )

    print("初期学習が完了しました。")


    # ---
    # # ステップ5: ファインチューニングの準備 (スコープ内へ移動)
    # ---
    print("\nステップ5: ファインチューニングの準備をします...")

    # ベースモデルの層を凍結解除し、ファインチューニングを行う
    base_model.trainable = True
    fine_tune_at = int(len(base_model.layers) * 0.3)  # ベースモデルの70%をファインチューニング対象にする
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=AdamW(learning_rate=1e-4, weight_decay=1e-5),  # AdamWオプティマイザを使用
        loss=losses.CategoricalCrossentropy(label_smoothing=0.1),  # ラベルスムージングを適用
        metrics=["accuracy"],
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    )


    # ---
    # # ステップ6: ファインチューニングの実行 (スコープ内へ移動)
    # ---
    print("\nステップ6: ファインチューニングを開始します...")
    fine_tune_epochs = 20
    total_epochs = initial_epochs + fine_tune_epochs

    # initial_epochを指定して、学習を再開
    history_fine = model.fit(
        train_pipeline,
        epochs=total_epochs,
        initial_epoch=history.epoch[-1] + 1,  # 前回の学習の続きからエポック数をカウント
        validation_data=validation_pipeline,
        callbacks=[reduce_lr],  # 学習率スケジューラをコールバックに追加
    )
    print("ファインチューニングが完了しました。")

# モデルの保存はスコープの外で行って問題ありません
model.save("models/food101_efficientnetv2b0_finetuned_final.keras")


# ---
# # ステップ7: 最終評価と学習曲線のプロット
# ---
print("\nステップ7: モデルの最終評価を行います...")
# 評価も分散戦略のスコープ内で行うことが推奨されますが、
# モデルがすでにコンパイルされているため、スコープ外でも動作する場合があります。
# より厳密には with strategy.scope(): の中に移動します。
loss, accuracy = model.evaluate(validation_pipeline)
print(f"\nファインチューニング後の検証データでの正解率 (Accuracy): {accuracy:.4f}")


# 2つの学習履歴を結合してプロット
def plot_fine_tune_history(original_history, fine_tune_history):
    acc = original_history.history["accuracy"] + fine_tune_history.history["accuracy"]
    val_acc = (
        original_history.history["val_accuracy"]
        + fine_tune_history.history["val_accuracy"]
    )
    loss = original_history.history["loss"] + fine_tune_history.history["loss"]
    val_loss = (
        original_history.history["val_loss"] + fine_tune_history.history["val_loss"]
    )

    epochs = range(len(acc))  # エポックの総数

    plt.figure(figsize=(14, 6)) # 全体の図のサイズを少し大きくしました

    # --- Training and Validation Accuracy プロット ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label="Training Accuracy", color='blue')
    plt.plot(epochs, val_acc, label="Validation Accuracy", color='orange')

    # 10エポックごとに点を打ち、数値をプロット
    for i in range(0, len(epochs), 10):
        # 訓練精度
        plt.plot(epochs[i], acc[i], 'o', color='blue', markersize=6)
        # テキスト位置を調整
        plt.text(epochs[i], acc[i] - 0.02, f'{acc[i]:.2f}', fontsize=9, ha='center', va='top', color='blue')

        # 検証精度
        plt.plot(epochs[i], val_acc[i], 'o', color='orange', markersize=6)
        # テキスト位置を調整
        plt.text(epochs[i], val_acc[i] + 0.02, f'{val_acc[i]:.2f}', fontsize=9, ha='center', va='bottom', color='orange')


    plt.plot(
        [initial_epochs - 1, initial_epochs - 1],
        plt.ylim(),
        label="Start Fine Tuning",
        linestyle="--",
        color='green' # 色を明示的に指定
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")
    plt.grid(True, linestyle=':', alpha=0.7) # グリッド線を追加

    # --- Training and Validation Loss プロット ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label="Training Loss", color='blue')
    plt.plot(epochs, val_loss, label="Validation Loss", color='orange')

    # 10エポックごとに点を打ち、数値をプロット
    for i in range(0, len(epochs), 10):
        # 訓練損失
        plt.plot(epochs[i], loss[i], 'o', color='blue', markersize=6)
        # テキスト位置を調整
        plt.text(epochs[i], loss[i] + 0.05, f'{loss[i]:.2f}', fontsize=9, ha='center', va='bottom', color='blue')

        # 検証損失
        plt.plot(epochs[i], val_loss[i], 'o', color='orange', markersize=6)
        # テキスト位置を調整
        plt.text(epochs[i], val_loss[i] - 0.05, f'{val_loss[i]:.2f}', fontsize=9, ha='center', va='top', color='orange')


    plt.plot(
        [initial_epochs - 1, initial_epochs - 1],
        plt.ylim(),
        label="Start Fine Tuning",
        linestyle="--",
        color='green' # 色を明示的に指定
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    plt.grid(True, linestyle=':', alpha=0.7) # グリッド線を追加

    plt.tight_layout() # レイアウトを調整
    plt.savefig("images/learning_curve_finetunedV2b0_final.png")
    plt.show() # 全ての描画が終わった後に一度だけ表示

plot_fine_tune_history(history, history_fine)


# ---
# # ステップ8: 検証データによる予測結果の可視化
# ---
print("\nステップ8: 予測結果の可視化を開始します...")

# クラス名のリストを取得
class_names = ds_info.features["label"].names


# 検証データセットから1バッチ分の画像とラベルを取得
# tf.data.Datasetをイテレートして、numpy配列として画像とラベルを取得
# ここで取得される画像は、preprocess_imgによってリサイズされているが、
# EfficientNetV2のpreprocess_inputはまだ適用されていない「生」のピクセル値を持つべきです。
raw_images_ds = validation_ds.map(preprocess_img, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)
images, labels = next(iter(raw_images_ds)) # 元の画像を取得
labels_to_plot_one_hot = tf.one_hot(labels, NUM_CLASSES).numpy() # ラベルはone-hotに変換


# モデルに入力するために、別途preprocess_inputを適用した画像を用意する
images_for_prediction = preprocess_input(tf.cast(images, tf.float32)).numpy()

# モデルによる予測を実行
predictions = model.predict(images_for_prediction)

# 結果をプロット
plt.figure(figsize=(15, 15))
for i in range(min(BATCH_SIZE, 25)):  # 最大25枚まで表示
    plt.subplot(5, 5, i + 1)

    # 画像を表示
    # tfds.loadされた画像は通常0-255のuint8形式なので、そのまま表示
    # もしfloat型で0-1に正規化されている場合は、*255.0してuint8にキャスト
    display_image = images[i].numpy().astype("uint8") # ここでnumpy配列に変換し、uint8型であることを確認

    plt.imshow(display_image)

    # 予測確率が最も高いクラスのインデックスを取得
    predicted_index = np.argmax(predictions[i])
    # 予測の信頼度（確率）を取得
    confidence = np.max(predictions[i])

    # 正解ラベル名と予測ラベル名を取得
    true_label_index = np.argmax(labels_to_plot_one_hot[i])
    true_label_name = class_names[true_label_index]
    predicted_label_name = class_names[predicted_index]

    # 正解なら青、不正解なら赤でタイトルを表示
    title_color = "blue" if predicted_index == true_label_index else "red"

    plt.title(
        f"True: {true_label_name}\nPred: {predicted_label_name} ({confidence:.2f})",
        color=title_color,
        fontsize=10,
    )
    plt.axis("off")

plt.tight_layout()
plt.savefig("images/prediction_resultsB0.png")
plt.show() # ここにもplt.show()を追加して、画像表示を保証します

print("予測結果の可視化が完了しました。")