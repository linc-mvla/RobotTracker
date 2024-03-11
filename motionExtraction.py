import cv2
import numpy as np
import traceback
import random
import json

#
sameThresh = 0.5
varianceThresh = 0.1

tClose = 3
tThresh = 7

rThresh = 10

frameDiff = 1
frameskip = 100
moveThresh = 40
robotThresh = 10
robotSize = 50
showSize = (540, 380)
prevFrames = []

maxContours = 15
#

video = cv2.VideoCapture(r'Final Tiebreaker - 2024 Silicon Valley Regional.mp4')
ret, frame = video.read()
rows, cols, ch = frame.shape

idAreas = np.zeros((rows, cols, 3), dtype = np.uint8) #(index base 255)
tAreas = np.zeros((rows, cols), dtype = np.float64)
paths = np.zeros((rows, cols, 3), dtype = np.uint8)
maxT = tThresh - tClose
class FieldObject():
    objNum = 1
    objects = []

    def __init__(self, initPoint, radius, t):
        self.id = FieldObject.objNum
        self.color = (self.id//(255*255), self.id//255, self.id % 255)
        self.randColor = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        FieldObject.objNum += 1

        self.path = []

        FieldObject.objects.append(self)
        self.addPoint(initPoint, radius, t)

    def addPoint(self, point, radius, t):
        if len(self.path) > 0:
            cv2.line(paths, self.path[-1][0], point, self.randColor, thickness = 2)

        cv2.circle(idAreas, point, r, self.color, cv2.FILLED)
        cv2.circle(tAreas, point, r, t, cv2.FILLED)
        self.path.append((point, radius, t))

    def getObject(point, radius, t):
        if radius < 5:
            return None
        
        area = idAreas[(point[1] - radius) : (point[1] + radius), (point[0] - radius) : (point[0] + radius)]

        if area.size == 0:
            return None

        size = area.size//3
        area = np.reshape(area, (size, 3))

        tArea =  tAreas[(point[1] - radius) : (point[1] + radius), (point[0] - radius) : (point[0] + radius)] - (t - tThresh)
        tArea[tArea < 0] = 0
        tArea[tArea > tClose] = tClose
        tArea = np.reshape(tArea, (size,))

        # tArea = 
        # area = area[(area).sum(axis = 1) > 0]

        # nonZero = area.size//3

        # # print(nonZero, size)
        # if nonZero/size < 0.1:
        #     return 0

        #average pixels within radius

        if np.sum(tArea) == 0:
            return 0

        avgCol = np.average(area, weights = tArea, axis = 0)
        stdCol = np.average((area - avgCol)**2, weights = tArea, axis = 0)

        avg = avgCol[0]*(255*255) + avgCol[1]*255 + avgCol[2]
        std = stdCol[0]*(255*255) + stdCol[1]*255 + stdCol[2]

        #print(avg, std)

        if std > varianceThresh:
            return None
        if 0.5 - abs((avg % 1.0) - 0.5) < sameThresh:
            id = round(avg)
            if id == 0:
                return 0
            
            o = FieldObject.objects[id - 1]
            p, r, t = o.path[-1]
            if abs(r - radius) > rThresh:
                return 0

            dx = point[0]-p[0]
            dy = point[1]-p[1]
            d = (dx*dx) + (dy*dy)
            if d < min(radius*radius, r*r):
                return o
            else:
                return None
        return None
        
objects = []

try:
    n = 0
    while video.isOpened():
        ret, frame = video.read()
        n += 1
        if not ret:
            break

        if n < frameskip:
            continue

        if len(prevFrames) < frameDiff:
            prevFrames.append(frame)
            continue

        prevFrame = prevFrames.pop(0)
        prevFrames.append(frame)

        #filter = cv2.subtract(frame, prevFrame)

        filter = cv2.absdiff(frame, prevFrame)

        gray = cv2.cvtColor(filter, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, moveThresh, 255, cv2.THRESH_BINARY)

        masked = cv2.bitwise_and(frame, filter, mask = mask)
        integral = np.float32(cv2.integral(mask))

        integral1 = cv2.warpAffine(integral, np.float32([[1,0,0],[0,1,robotSize]]), (cols + 1, rows + 1))
        integral2 = cv2.warpAffine(integral, np.float32([[1,0,robotSize],[0,1,0]]), (cols + 1, rows + 1))
        integral3 = cv2.warpAffine(integral, np.float32([[1,0,robotSize],[0,1,robotSize]]), (cols + 1, rows + 1))
        vals = integral - integral1 - integral2 + integral3

        diff = (vals/(robotSize*robotSize)).astype('uint8')
        diff = cv2.warpAffine(diff, np.float32([[1,0,-robotSize/2.0],[0,1,-robotSize/2.0]]), (cols, rows))

        _, robMask = cv2.threshold(diff, robotThresh, 255, cv2.THRESH_BINARY)

        contours, heirarchy = cv2.findContours(image = robMask, mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_NONE)

        #out = cv2.bitwise_and(frame, frame, mask = robMask)
        #cv2.drawContours(image = out, contours = contours, contourIdx = -1, color = (0,255,0), thickness = 2)

        if len(contours) > maxContours:
            continue

        out = idAreas.copy()

        cv2.addWeighted(paths, 1, out, 1, 0, out)
        #out[maskLine] = paths

        for c  in contours:
            if len(c) > 1000:
                continue

            center = cv2.mean(c)[:2]
            center = (int(center[0]), int(center[1]))
            dist = (c-center)
            r = int(np.mean(np.sqrt((dist*dist).sum(axis = 2))))

            obj = FieldObject.getObject(center, r, n)
            if obj is None:
                pass
            elif obj == 0:
                FieldObject(center, r, n)
            else:
                obj.addPoint(center, r, n)

            cv2.circle(out, center, r, (0,255,0), thickness = 2)

        out = cv2.resize(out, showSize, fx = 0, fy = 0, interpolation = cv2.INTER_CUBIC)
        frame = cv2.resize(frame, showSize, fx = 100, fy = 0, interpolation = cv2.INTER_CUBIC)
        cv2.imshow("original", frame)
        cv2.imshow("filter", out)

        if (cv2.waitKey(25) & 0xFF) == ord('q'):
            break
    
except Exception as error:
    traceback.print_exc()
    pass
finally:
    cv2.imwrite("paths.png", paths) 
    video.release()
    cv2.destroyAllWindows()
    with open("paths.json", "w") as file:
        json_str = json.dumps({p.id : p.path for p in FieldObject.objects})
        json_str = json_str.replace(', "', ',\n "')
        file.write(json_str)
        #json.dump(json.load()
