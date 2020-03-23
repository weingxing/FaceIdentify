import cv2

capture = cv2.VideoCapture()
flag = capture.open(0)
# capture.release()
capture.set(cv2.CAP_PROP_BRIGHTNESS, 60)
capture.set(cv2.CAP_PROP_FPS, 30)
print(capture.get(cv2.CAP_PROP_BRIGHTNESS))
while True:
    flag, frame = capture.read()
    cv2.imshow('out', frame)
    # Press Q on keyboard to stop recording
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
capture.release()
