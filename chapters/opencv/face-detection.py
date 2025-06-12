import cv2
import matplotlib.pyplot as plt


def main():
    # 画像の読み込み
    ramen_mans = cv2.imread("images/opencv/ramen_mans.jpg")

    # RGB 変換
    img = cv2.cvtColor(ramen_mans, cv2.COLOR_BGR2RGB)

    # 事前にアップロードしたファイル読み込み
    cascade = cv2.CascadeClassifier("utils/haarcascade_frontalface_default.xml")

    # 顔を検出
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
    )

    # 検出した領域を矩形で囲む
    for x, y, w, h in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), (200, 0, 0), 3)

    # 画像の保存
    cv2.imwrite("images/ramen_mans_face_detection.jpg", img)

    # 画像の表示
    plt.imshow(img)
    plt.axis("off")
    plt.title("Face Detection")
    plt.show()


if __name__ == "__main__":
    main()
