import cv2
import matplotlib.pyplot as plt


def main():
    # 画像の読み込み
    img = cv2.imread("images/cheer_man.jpg")
    cvt_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 画像の表示
    plt.imshow(cvt_img)
    plt.show()


if __name__ == "__main__":
    main()
