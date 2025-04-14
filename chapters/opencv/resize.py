import cv2
import matplotlib.pyplot as plt


def main():
    # 画像のリサイズ
    img = cv2.imread("images/test.png")
    resized_img = cv2.resize(img, (200, 200))

    # 倍率でリサイズ
    img2 = cv2.imread("images/cheer_man.jpg")
    resized_img2 = cv2.resize(img2, None, fx=0.5, fy=0.5)

    # リサイズした画像の保存
    cv2.imwrite("images/test_resized.png", resized_img)
    cv2.imwrite("images/cheer_man_resized.jpg", resized_img2)

    # リサイズした画像の表示
    # cvt_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    cvt_img2 = cv2.cvtColor(resized_img2, cv2.COLOR_BGR2RGB)
    # plt.imshow(cvt_img)

    plt.imshow(cvt_img2)
    plt.show()


if __name__ == "__main__":
    main()
