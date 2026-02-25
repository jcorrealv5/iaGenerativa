import cv2

video = cv2.VideoCapture(0)
if(video.isOpened()):
    while(True):
        rpta, img = video.read()
        if(rpta):
            cv2.imshow("Video", img)
            key = cv2.waitKey(1)
            if(key==ord("s")):
                break
        else:
            break
    video.release()
    cv2.destroyAllWindows()
else:
    print("No esta activa la camara")