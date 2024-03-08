from PIL import Image
import cv2

video = cv2.VideoCapture(r'video2.mp4')

try:
    frameDiff = 1
    prevFrames = []
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        if len(prevFrames) < frameDiff:
            prevFrames.append(frame)
            continue

        prevFrame = prevFrames.pop(0)
        prevFrames.append(frame)

        #filter = cv2.subtract(frame, prevFrame)

        filter = cv2.absdiff(frame, prevFrame)

        filter = cv2.resize(filter, (540, 380), fx = 0, fy = 0, interpolation = cv2.INTER_CUBIC)
        #cv2.imshow('Frame', frame)
        cv2.imshow("filter", filter)

        if (cv2.waitKey(25) & 0xFF) == ord('q'):
            break
except:
    pass

video.release()
cv2.destroyAllWindows()

# newVideo = None
# frameDiff = offset * fps
# for i in range(0, numFrames - frameDiff):
#     frame = video[i] - video[i+frameDiff]
#     newVideo.add(frame)

# newVideo.save()
