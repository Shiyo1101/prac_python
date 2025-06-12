import cv2
import matplotlib.pyplot as plt
import numpy as np


def main():
    # 画像の読み込み
    img = cv2.imread("images/opencv/cheer_man.jpg")

    # グレースケール変換
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 輪郭を抽出するための設定
    ret, thresh = cv2.threshold(img_gray, 150, 255, 0)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    cnt_img = np.zeros_like(thresh, dtype=np.uint8)
    cnt_img = cv2.drawContours(cnt_img, contours, -1, 255, 2)

    # 画像の保存
    cv2.imwrite("images/cheer_man_contour_extraction.jpg", cnt_img)

    # 画像の表示
    plt.imshow(cnt_img, cmap="gray")
    plt.axis("off")
    plt.title("Contour Extraction")
    plt.show()


if __name__ == "__main__":
    main()
