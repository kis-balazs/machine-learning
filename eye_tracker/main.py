import cv2

# to detect the face and eyes to detect the face
faces = cv2.CascadeClassifier("face.xml")
eyes = cv2.CascadeClassifier("eye.xml")

# capture the frame through webcam
capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # (_, blackAndWhiteImage) = cv2.threshold(gray_frame, 127, 255, cv2.THRESH_BINARY)

    face = faces.detectMultiScale(gray_frame, 1.3, 5)
    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), thickness=4)

        # for i in gray_frame.height:
        #     for j in gray_frame.width:
        #         if (i < x+w and i > x) and (j < y+h and j > y):
        #             print(gray_frame[j, i])
        gray_face = gray_frame[y:y + h, x:x + w]
        color_face = frame[y:y + h, x:x + w]
        # cv2.imshow('', gray_face)

        eye = eyes.detectMultiScale(gray_face, 1.3, 10)
        i = 0
        for (eyeX, eyeY, eyeW, eyeH) in eye:
            _, eyeCandidate = cv2.threshold(gray_face[eyeY:eyeY + eyeH, eyeX:eyeX + eyeW], 127, 255, cv2.THRESH_BINARY)
            # cv2.imshow('i' + str(i), eyeCandidate)
            i += 1
            whiteRatio = 1
            for i in range(0, eyeCandidate.shape[0]):
                for j in range(0, eyeCandidate.shape[1]):
                    if eyeCandidate[i, j] == 255:
                        whiteRatio += 1
            whiteRatio = eyeCandidate.shape[0] * eyeCandidate.shape[1] / whiteRatio
            if whiteRatio > 5:
                cv2.rectangle(color_face, (eyeX, eyeY), (eyeX + eyeW, eyeY + eyeH), (0, 255, 0), thickness=4)

    cv2.imshow("frame", frame)
    cv2.imshow("gray_frame", gray_frame)
    if cv2.waitKey(1) == 13:
        break

capture.release()
cv2.destroyAllWindows()
