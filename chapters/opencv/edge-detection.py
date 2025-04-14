import cv2
import matplotlib.pyplot as plt
import numpy as np


def main():
    # 画像の読み込み
    img = cv2.imread("images/ramen_man.jpg")

    # 画像のグレースケール化
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sobel フィルタを適用してエッジを検出
    sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=5)

    # sobel_x と sobel_y を絶対値に変換してから合成する
    sobel_combined = cv2.addWeighted(
        np.absolute(sobel_x), 0.5, np.absolute(sobel_y), 0.5, 0
    )

    # 画像の保存
    cv2.imwrite("images/ramen_man_sobel.jpg", sobel_combined)

    # 画像の表示
    plt.imshow(sobel_combined, cmap="gray")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
