import cv2

cap = cv2.VideoCapture("ball.mp4")

#Object detection from stable camera

object_detector = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()

    #Object Detection
    mask = object_detector.apply(frame)
    contours, _ = cv2.findContours(mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            #cv2.drawContours(frame, [cnt], - 1, (0, 255, 0), 2)
            try:
                
                x, y, w, h = cv2.boundingRect(cnt)
                print(str(x) +  "    " + str(y))
                #cv2.circle(frame, (x + w/2, y + w/2), w/2, 0, 200)
            except:
                print()



    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(30)
    if key == 27:   # ESC key
        break

cap.release()
cv2.destroyAllWindows()