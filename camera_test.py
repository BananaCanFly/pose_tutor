import cv2

for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        print(f"Camera {i} opened")
        ret, frame = cap.read()
        if ret:
            cv2.imshow(f"cam {i}", frame)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
        cap.release()
