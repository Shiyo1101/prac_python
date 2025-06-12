import cv2
import matplotlib.pyplot as plt


def main():
    threshold = 125

    # 画像の読み込み
    img = cv2.imread("images/opencv/ramen_man.jpg")

    # 画像のグレースケール化
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 画像の二値化(閾値 125 を超えた画素を255にする。)
    ret, img_thresh = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)

    # 画像の保存
    cv2.imwrite("images/ramen_man_thresh.jpg", img_thresh)

    # 画像の表示
    plt.subplot(121)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.subplot(122)
    plt.title("Threshold")
    plt.imshow(cv2.cvtColor(img_thresh, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
