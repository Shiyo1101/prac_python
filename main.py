import cv2
import matplotlib.pyplot as plt


def main():
    # png to jpg
    img = cv2.imread("test.png")
    cv2.imwrite("test.jpg", img)

    jpgImg = cv2.imread("test.jpg")
    plt.imshow(jpgImg)

    plt.show()


if __name__ == "__main__":
    main()
