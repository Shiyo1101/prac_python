import tensorflow as tf
from keras.api.preprocessing import image_dataset_from_directory

# パスを指定（catとdogが入っている親ディレクトリ）
data_dir = "dataset/cats_and_dogs"  # データセットのパスを指定

# データセットを80%訓練、20%テストに分割
batch_size = 32
img_height = 128
img_width = 128
validation_split = 0.2

# トレーニングデータセットとバリデーションデータセットを作成
train_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

val_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

# クラス名を取得
class_names = train_ds.class_names

# データセットの前処理
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
