import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

filename = "Huenmemepaths.json"
minLength = 20

with open(filename, "r") as file:
    data = json.load(file)

width = 1280
height = 720

addP = lambda p1, p2 : (p1[0] + p2[0], p1[1] + p2[1])
subP = lambda p1, p2 : (p1[0] - p2[0], p1[1] - p2[1])
divP = lambda p, k : (p[0]/k, p[1]/k)
mulP = lambda p, k : (p[0]*k, p[1]*k)
dotP = lambda p1, p2 : p1[0]*p2[0] + p1[1]*p2[1]
pMagn = lambda p : (p[0]*p[0] + p[1]*p[1])**(0.5)

trapezoid = [
    (409, 189),
    (850, 188),
    (1100, 372),
    (152, 372)
]

tTop = subP(trapezoid[1], trapezoid[0])[0]
tBot = subP(trapezoid[2], trapezoid[3])[0]
tHeight = subP(trapezoid[0], trapezoid[3])[1]

fieldWidth = 16.5
fieldHeight = 8.25


count = 0
newPaths = []
tMax = 0
for id, path in data.items():
    if len(path) < 5:
        continue

    cx = [p[0][0] for p in path]
    cy = [p[0][1] for p in path]
    dx = max(cx) - min(cx)
    dy = max(cy) - min(cy)
    if dx*dx + dy*dy < minLength*minLength:
        continue

    x = []
    y = []
    newPath = []
    for p, r, t in path:
        p2 = subP(p, trapezoid[3])
        yScale = fieldHeight/tHeight
        ry = fieldHeight - (p2[1] + r*0.3) * yScale #offset robot by a bit of the radius
        percent = (tHeight - ry)/tHeight
        xScale = fieldWidth / (percent*tBot + (1.0-percent)*tTop)
    
        rx = p2[0] * xScale

        if t > tMax:
            tMax = t

        if rx>fieldWidth or rx<0 or ry>fieldHeight or ry<0:
            continue

        x.append(p[0])
        y.append(p[1])
        
        newPath.append(((rx,ry), t))

    color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
    newPaths.append((newPath, color))
    plt.plot(x,y)

    count += 1

plt.show()

t = 0

video = cv2.VideoCapture(r'Match 1 (R1) - 2024 Hueneme Port Regional.mp4')

field = cv2.imread("2024Game2.png") 
imgHeight, imgWidth, _ = field.shape
scale = imgWidth/fieldWidth

timeDiff = 30
timeSkip = 200
while t < tMax:
    ret, frame = video.read()
    t += 1
    if t < timeSkip:
        continue

    paths = np.zeros((imgHeight, imgWidth, 3), dtype=np.uint8)
    for path, color in newPaths:
        if len(path) == 0:
            continue
        lastP = mulP(path[0][0],scale)
        lastt = path[0][1]
        for point, pt in path[1:]:
            point = mulP(point, scale)
            if pt > t and lastt > t:
                break

            if pt > t-timeDiff or lastt > t-timeDiff:
                diff = subP(point, lastP)
                dt = pt - lastt
                t1 = (max(lastt, t - timeDiff) - lastt) / dt
                t2 = (min(pt, t) - lastt) / dt

                p1 = addP(lastP, mulP(diff, t1))
                p2 = addP(lastP, mulP(diff, t2))

                p1i = (int(p1[0]), int(p1[1]))
                p2i = (int(p2[0]), int(p2[1]))

                if pMagn(subP(p1, p2)) < 0.5:
                    cv2.circle(paths, p1i, 6, color, thickness = cv2.FILLED)
                else:
                    cv2.line(paths, p1i, p2i, color, thickness = 2)

            lastP = point
            lastt = pt
    
    
    frame = cv2.resize(frame, (540, 380), fx = 100, fy = 0, interpolation = cv2.INTER_CUBIC)

    cv2.imshow("paths", cv2.addWeighted(paths, 1, field, 1, 0))
    cv2.imshow("video", frame)

    if (cv2.waitKey(25) & 0xFF) == ord('q'):
        break

cv2.destroyAllWindows()