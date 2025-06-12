import cv2
import matplotlib.pyplot as plt


def main():
    # 画像の読み込み
    img = cv2.imread("images/opencv/test.png")

    # 画像の回転
    img_rotate = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    # 画像の保存
    cv2.imwrite("images/test_rotate.png", img_rotate)

    # 画像の表示
    plt.imshow(cv2.cvtColor(img_rotate, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
