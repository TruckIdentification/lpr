#coding=utf-8

import os
from lpr import pipline as pp
import cv2
import numpy as np
import tensorflow as tf
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

camera = cv2.VideoCapture('cars.mp4')
fontC = ImageFont.truetype("./Font/platech.ttf", 70, 0)
#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#videoWriter = cv2.VideoWriter('gaiplate.mp4',fourcc,20,(1280,720))
# 帧率
fps = 5
# 总是取前一帧做为背景（不用考虑环境影响）
pre_frame = None
white = 0
black = 0
n = 0
m = 1000
path = '/home/cocoyj/PycharmProjects/chepaishibie/car_tests/'
plate = ''
while (1):

    ret, frame = camera.read()
    if ret ==None:
        break
    # 转灰度图
    cv2.rectangle(frame,(480,0),(1280,480),(255,0,0),2)
    frame_cap = frame[0:480,480:1200]
    frame_out = frame_cap
    frame_cap = cv2.resize(frame_cap, (150, 90))
    gray_lwpCV = cv2.cvtColor(frame_cap, cv2.COLOR_BGR2GRAY)

    

    gray_lwpCV = cv2.GaussianBlur(gray_lwpCV, (21, 21), 0)

    # 如果没有背景图像就将当前帧当作背景图片
    if pre_frame is None:
        pre_frame = gray_lwpCV
    else:
        # absdiff把两幅图的差的绝对值输出到另一幅图上面来
        img_delta = cv2.absdiff(pre_frame, gray_lwpCV)
        # threshold阈值函数(原图像应该是灰度图,对像素值进行分类的阈值,当像素值高于（有时是小于）阈值时应该被赋予的新的像素值,阈值方法)
        thresh = cv2.threshold(img_delta, 25, 255, cv2.THRESH_BINARY)[1]
        # 膨胀图像
        thresh = cv2.dilate(thresh, None, iterations=2)

        for i in range(thresh.shape[0]):
            for j in range(thresh.shape[1]):
                if thresh[i][j] == 0:
                    black+=1
                else:
                    white+=1

        if white/13500 > 0.0:
            n+=1
            if n==5:
                path_out = path+str(m)+'plate.jpg'
                cv2.imwrite(path_out,frame)
                frame = cv2.imread(path_out)
                frame, res = pp.SimpleRecognizePlate(frame)
                if res ==[]:
                    plate=''
                else:
                    plate = res[0]
                m+=1
        else:
            n=0
        black = white = 0
        pre_frame = gray_lwpCV
    # frame = cv2.putText(frame, '中文'.encode('utf-8'), (800, 800), 2, 3, (255, 0, 0), 2)

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    # draw.text((int(rect[0]+1), int(rect[1]-16)), addText.decode("utf-8"), (255, 255, 255), font=fontC)
    draw.text((600, 600), plate, (255, 255, 255), font=fontC)
    imagex = np.array(img)
    cv2.imshow("capture", imagex)
    #videoWriter.write(imagex)
        # cv2.imshow('1',thresh)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

    # release()释放摄像头
camera.release()
#videoWriter.release()
# destroyAllWindows()关闭所有图像窗口
cv2.destroyAllWindows()


