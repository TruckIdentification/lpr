import cv2
import numpy as np
import math
import rotate_test as rt



def hough(img_c):
    img_c = cv2.GaussianBlur(img_c,(3,3),0)
    # gray = cv2.cvtColor(img_c,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img_c,50,150,apertureSize=3)
    # cv2.imshow('edges',edges)
    minLineLength = 50  #线段长度阈值,表示最低线段的长度，比这个设定参数短的线段就不能被显现出来
    maxLineGap = 20     #线段上最近两点之间的阈值,有默认值0，允许将同一行点与点之间连接起来的最大的距离
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
    for x1,y1,x2,y2 in lines[0]:
        cv2.line(img_c,(x1,y1),(x2,y2),(0,255,0),2)
        k = math.tan(float((y1-y2)/(x1-x2)))*180/(math.pi)
    return k

# asd = cv2.imread('111.png')
# n = hough(asd)
# print(n)