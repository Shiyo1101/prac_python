import cv2
import matplotlib.pyplot as plt


def main():
    # 2枚の画像を読み込む
    img1 = cv2.imread("images/opencv/ramen_man.jpg", 0)
    img2 = cv2.imread("images/opencv/cheer_man.jpg", 0)

    # 特徴量量検出
    akaze = cv2.AKAZE_create()
    kp1, des1 = akaze.detectAndCompute(img1, None)
    kp2, des2 = akaze.detectAndCompute(img2, None)

    # 特徴量マッチング
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # 特徴点間のハミング距離でソート
    matches = sorted(matches, key=lambda x: x.distance)

    # 2 つの画像のマッチング結果を作成、最も似ている 5 箇所を表示する
    img1_2 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:5], None)

    # 画像を保存
    cv2.imwrite("images/matching.jpg", img1_2)

    # 画像を表示
    plt.imshow(cv2.cvtColor(img1_2, cv2.COLOR_BGR2RGB))
    plt.title("Feature Matching")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
