import cv2
import matplotlib.pyplot as plt


def main():
    # 画像の読み込み
    img = cv2.imread("images/test.jpg")

    # グレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # グレースケール画像の保存
    cv2.imwrite("images/test_gray.jpg", gray)

    # グレースケール画像の表示
    plt.imshow(gray, cmap="gray")
    plt.axis("off")  # 軸を非表示にする
    plt.show()


if __name__ == "__main__":
    main()
