import cv2
import numpy as np
import traceback
import random
import json

# Constants
# sameThresh = 0.5
# varianceThresh = 500

percentMatch = 0.7 #if area covered is some percent
percentUnknown = 0.3 # if area is covered by multiple objects

tClose = 1 * 30  #      ________ 0   
tThresh = 2 * 30 #     / tClose
                 #____/tThresh 

rThresh = 10 # threshold on changing radius

frameDiff = 1
frameskip = 200

moveThresh = 40
robotThresh = 30
robotSize = 45

showSize = (540, 380)
prevFrames = []

maxContours = 15

#motion, average, contours, time, tracking
mode = 'time'

fileName = 'Huenmeme'
video = cv2.VideoCapture(r'Match 1 (R1) - 2024 Hueneme Port Regional.mp4')

ret, frame = video.read()
rows, cols, ch = frame.shape

idAreas = np.zeros((rows, cols), dtype = np.float64)
tAreas = np.zeros((rows, cols), dtype = np.float64)
paths = np.zeros((rows, cols, 3), dtype = np.uint8)

class FieldObject():
    objNum = 1
    objects = []

    def __init__(self, initPoint, radius, t):
        self.id = FieldObject.objNum
        self.randColor = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        FieldObject.objNum += 1

        self.path = []

        FieldObject.objects.append(self)
        self.addPoint(initPoint, radius, t)

    def addPoint(self, point, radius, t):
        if len(self.path) > 0:
            cv2.line(paths, self.path[-1][0], point, self.randColor, thickness = 2)

        cv2.circle(idAreas, point, r, self.id, cv2.FILLED)
        cv2.circle(tAreas, point, r, t, cv2.FILLED)
        self.path.append((point, radius, t))

    def updateT(point, r, t):
        cv2.circle(tAreas, point, r, t, cv2.FILLED)

    def getObject(point, radius, t):
        if radius < 5:
            return None
        
        area = idAreas[(point[1] - radius) : (point[1] + radius), (point[0] - radius) : (point[0] + radius)].copy()

        if area.size == 0:
            return None
        
        area = np.reshape(area, (area.size,))

        if np.sum(area) == 0:
            return 0

        tArea = tAreas[(point[1] - radius) : (point[1] + radius), (point[0] - radius) : (point[0] + radius)].copy() - (t - tThresh)
        tArea[tArea < 0] = 0
        tArea[tArea > (tThresh - tClose)] = tThresh - tClose
        tArea = np.reshape(tArea, (tArea.size,))

        if np.sum(tArea) == 0:
            return 0

        ids, inv = np.unique(area, return_inverse = True)
        counts = np.bincount(inv, tArea)

        totCounts = tArea.size * (tThresh - tClose)
        maxPercent = 0
        for id, n in zip(ids, counts):
            percent = n / totCounts
            if percent > percentMatch:
                if id == 0:
                    return 0
                o = FieldObject.objects[int(id) - 1]
                p, r, t = o.path[-1]
                if abs(r - radius) < rThresh:
                    dx = point[0]-p[0]
                    dy = point[1]-p[1]
                    d = (dx*dx) + (dy*dy)
                    if d < min(radius*radius, r*r):
                        return o
            elif percent > maxPercent:
                maxPercent = percent
                
        if maxPercent > percentUnknown:
            return -1

        # #average pixels within radius
        # avg = np.average(area, weights = tArea, axis = 0)
        # std = np.average((area - avg)**2, weights = tArea, axis = 0)

        # #print(avg, std)

        # if std > varianceThresh:
        #     return -1
        # if 0.5 - abs((avg % 1.0) - 0.5) < sameThresh:
        #     id = round(avg)
        #     if id == 0:
        #         return 0
            
        #     o = FieldObject.objects[id - 1]
        #     p, r, t = o.path[-1]
        #     if abs(r - radius) > rThresh:
        #         return 0

        #     dx = point[0]-p[0]
        #     dy = point[1]-p[1]
        #     d = (dx*dx) + (dy*dy)
        #     if d < min(radius*radius, r*r):
        #         return o
        #     else:
        #         return None
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

        if len(contours) > maxContours:
            continue

        if mode == 'motion':
            out = masked
        elif mode == 'average':
            out = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
        elif mode == 'contour':
            out = cv2.cvtColor(robMask, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(image = out, contours = contours, contourIdx = -1, color = (0,255,0), thickness = 2)
        elif mode == 'time':
            # if np.max(tAreas) < 1:
            #     out = cv2.cvtColor(np.uint8(tAreas), cv2.COLOR_GRAY2BGR)
            # else:
            #     out = cv2.cvtColor(np.uint8(tAreas * (255.0 / np.max(tAreas))), cv2.COLOR_GRAY2BGR)
            tArea = (tAreas.copy() - (n - tThresh)) * (255.0 / (tThresh - tClose))
            tArea[tArea < 0] = 0
            tArea[tArea > 255] = 255
            out = cv2.cvtColor(np.uint8(tArea), cv2.COLOR_GRAY2BGR)
        else:
            if np.max(idAreas) < 1:
                out = cv2.cvtColor(np.uint8(idAreas), cv2.COLOR_GRAY2BGR)
            else:
                out = cv2.cvtColor(np.uint8(idAreas * (255.0 / np.max(idAreas))), cv2.COLOR_GRAY2BGR)

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
            color = (0, 255, 0)
            if obj is None:
                pass
            elif obj == -1:
                FieldObject.updateT(center, r, n)
                color = (0, 255, 255)
            elif obj == 0:
                FieldObject(center, r, n)
                color = (255, 0, 0)
            else:
                color = (0, 255, 0)
                obj.addPoint(center, r, n)

            cv2.circle(out, center, r, color, thickness = 2)

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
    cv2.imwrite(fileName+"Paths.png", paths) 
    video.release()
    cv2.destroyAllWindows()
    with open(fileName + "Paths.json", "w") as file:
        json_str = json.dumps({p.id : p.path for p in FieldObject.objects})
        json_str = json_str.replace(', "', ',\n "')
        file.write(json_str)
        #json.dump(json.load()
