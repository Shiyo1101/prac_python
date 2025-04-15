import cv2  # OpenCVライブラリをインポート


def main():
    # カスケード分類器を読み込む
    # Haar特徴分類器を使用して顔検出を行うための事前学習済みモデルを読み込む
    face_cascade = cv2.CascadeClassifier("utils/haarcascade_frontalface_default.xml")

    # ウェブカメラを開く
    # VideoCapture(1)はデバイスID 1のカメラを開く（通常0がデフォルトカメラ）
    # デバイスIDは環境によって異なる場合があるので、必要に応じて変更
    cap = cv2.VideoCapture(1)

    # カメラが開けない場合はエラーメッセージを表示して終了
    if not cap.isOpened():
        print("Error: Could not open camera.")  # カメラが開けない場合のエラー表示
        return

    print("Press 'q' to quit.")  # プログラム終了のためのキー操作を案内

    while True:
        # カメラからフレームを取得
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")  # フレーム取得に失敗した場合のエラー表示
            break

        # フレームをグレースケール画像に変換
        # 顔検出はグレースケール画像で行う方が効率的
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 顔を検出する
        # detectMultiScaleは画像内の顔を検出し、矩形領域（x, y, w, h）を返す
        # 引数: スケールファクター(1.1)、最小近傍矩形数(4)
        faces = face_cascade.detectMultiScale(gray, 1.1, 8)

        # 検出された顔ごとに処理を行う
        for x, y, w, h in faces:
            # 顔を囲む矩形を描画する
            # 矩形の色は青 (255, 0, 0)、線の太さは2
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # フレームをウィンドウに表示
        # ウィンドウ名は "FaceDetectionCamera"
        cv2.imshow("FaceDetectionCamera", frame)

        # ‘q’キーが押されたらループを終了する
        # waitKey(1)は1ミリ秒待機し、キー入力を受け付ける
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # カメラを解放する
    # カメラデバイスを閉じてリソースを解放
    cap.release()

    # すべてのOpenCVウィンドウを閉じる
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
