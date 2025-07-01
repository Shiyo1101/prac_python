# モデルの予測と実際のベータ分布の比較を行う

import matplotlib.pyplot as plt
import numpy as np
from keras.api.models import load_model
from scipy import stats

# モデルの読み込み
model = load_model("models/data-analytics/beta_model_v2.keras")

# テストデータの設定
a, b = 5, 3
x = np.linspace(0, 1, 1000)
y = stats.beta.pdf(x, a, b)

# モデルの予測
pred = model.predict(np.array([y]))
pred_a, pred_b = pred[0][0], pred[0][1]
pred_y = stats.beta.pdf(x, pred_a, pred_b)

# 予測結果の表示（グラフを重ねる）
plt.plot(x, y, label=f"Actual Beta(a={a}, b={b})", color="blue")
plt.plot(
    x, pred_y, label=f"Predicted Beta(a={pred_a:.2f}, b={pred_b:.2f})", color="red"
)
plt.title("Beta Distribution Comparison")
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.legend()
plt.grid()
plt.show()
