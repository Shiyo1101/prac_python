import cv2
import matplotlib.pyplot as plt


def main():
    # png to jpg
    img = cv2.imread("images/opencv/test.png")
    cv2.imwrite("images/test.jpg", img)

    jpgImg = cv2.imread("images/opencv/test.jpg")
    plt.imshow(jpgImg)

    plt.show()


if __name__ == "__main__":
    main()
