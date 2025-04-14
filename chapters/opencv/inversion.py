import cv2
import matplotlib.pyplot as plt


def main():
    # 画像の読み込み
    img = cv2.imread("images/test.png")

    # 画像の反転
    img_inversion = cv2.flip(img, 0)  # 0: 垂直反転, 1: 水平反転, -1: 両方反転

    # 画像の保存
    cv2.imwrite("images/test_inversion.png", img_inversion)

    # 画像の表示
    plt.imshow(cv2.cvtColor(img_inversion, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
