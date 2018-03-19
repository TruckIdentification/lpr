import cv2
import numpy as np
'''1、将采集到的彩色车牌图像转换成灰度图
2、灰度化的图像利用高斯平滑处理后，再对其进行中直滤波
3、使用Sobel算子对图像进行边缘检测
4、对二值化的图像进行腐蚀，膨胀，开运算，闭运算的形态学组合变换
5、对形态学变换后的图像进行轮廓查找，根据车牌的长宽比提取车牌'''
minPlateRatio = 2.5 # 车牌最小比例
maxPlateRatio = 5   # 车牌最大比例

# 图像处理
def imageProcess(gray):
    # 高斯平滑
    gaussian = cv2.GaussianBlur(gray, (3, 3), 0, 0, cv2.BORDER_DEFAULT)

    # Sobel算子，X方向求梯度
    sobel = cv2.convertScaleAbs(cv2.Sobel(gaussian, cv2.CV_16S, 1, 0, ksize=3))
    cv2.imwrite('sobel.jpg',sobel)
    # 二值化
    ret, binary = cv2.threshold(sobel, 150, 255, cv2.THRESH_BINARY)
    cv2.imwrite('binary.jpg',binary)
    # 对二值化后的图像进行闭操作
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (36, 8))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, element)
    cv2.imwrite('closed.jpg',closed )
    # 再通过腐蚀->膨胀 去掉比较小的噪点
    erosion = cv2.erode(closed, None, iterations=4)
    dilation = cv2.dilate(erosion, None, iterations=4)
    cv2.imwrite('dilation.jpg',dilation )
    # 返回最终图像
    return dilation

# 找到符合车牌形状的矩形
def findPlateNumberRegion(img):
    region = []
    # 查找外框轮廓
    contours_img, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print("contours lenth is :%s" % (len(contours)))
    # 筛选面积小的
    for i in range(len(contours)):
        cnt = contours[i]
        print('cnt=',cnt)
        #print(cnt)
        # 计算轮廓面积
        area = cv2.contourArea(cnt)
        print('area=',area)
        # 面积小的忽略
        if area < 2000:
            continue

        # 转换成对应的矩形（最小）
        rect = cv2.minAreaRect(cnt)
        print("rect is:%s" % {rect})

        # 根据矩形转成box类型，并int化
        box = np.int32(cv2.boxPoints(rect))
        print('box=',box)
        # 计算高和宽
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])
        # 正常情况车牌长高比在2.7-5之间,那种两行的有可能小于2.5，这里不考虑
        ratio = float(width) / float(height)
        if ratio > maxPlateRatio or ratio < minPlateRatio:
            continue
        # 符合条件，加入到轮廓集合
        region.append(box)
    return region

def detect(img):
  # 转化成灰度图
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # 形态学变换的处理
  dilation = imageProcess(gray)
  cv2.imwrite('imageprocess.jpg', dilation)  
# 查找车牌区域
  region = findPlateNumberRegion(dilation)
  # 默认取第一个
  print(region);
  print("region lenth is :%s" % (len(region)))
  box = region[0]
  print('last box=',box)
  #在原图画出轮廓
#   cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
  cv2.imwrite('drawContours.jpg', img)
  # 找出box四个角的x点，y点，构成数组并排序
  ys = [box[0, 1], box[1, 1], box[2, 1], box[3, 1]]
  xs = [box[0, 0], box[1, 0], box[2, 0], box[3, 0]]
  ys_sorted_index = np.argsort(ys)
  xs_sorted_index = np.argsort(xs)
  # 取最小的x，y 和最大的x，y 构成切割矩形对角线
  min_x = box[xs_sorted_index[0], 0]
  max_x = box[xs_sorted_index[3], 0]
  min_y = box[ys_sorted_index[0], 1]
  max_y = box[ys_sorted_index[3], 1]

  # 切割图片，其实就是取图片二维数组的在x、y维度上的最小minX,minY 到最大maxX,maxY区间的子数组
  img_plate = img[min_y:max_y, min_x:max_x]
  return img_plate


if __name__ == '__main__':
        imagePath = '粤B594SB.jpg' # 图片路径
        img = cv2.imread(imagePath)
        img = detect(img)
        cv2.imwrite('111.png', img)
        #cv2.imshow("img",img)
        #cv2.waitKey(0)
