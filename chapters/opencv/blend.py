import cv2
import matplotlib.pyplot as plt


def main():
    # 合成する画像を読み込む
    img = cv2.imread("images/opencv/cheer_man.jpg")
    logo = cv2.imread("images/opencv/test.png")

    # 合成する画像のサイズを取得
    h, w, _ = img.shape

    # ロゴ画像のサイズを合成する画像のサイズに合わせる
    logo = cv2.resize(logo, (w, h))

    # 画像の合成
    blended = cv2.addWeighted(src1=img, alpha=0.7, src2=logo, beta=0.3, gamma=0)

    # 合成した画像を保存
    cv2.imwrite("images/blended_image.jpg", blended)

    # 合成した画像を表示
    plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    plt.show()


if __name__ == "__main__":
    main()
