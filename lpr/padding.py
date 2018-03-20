import cv2
import numpy as np
#按照指定图像大小调整尺寸
# http://blog.csdn.net/u012319493/article/details/75937903
def resize_image(image, height, width):
    top, bottom, left, right = (0, 0, 0, 0)
    #获取图像尺寸
    h, w, l = image.shape

    #对于长宽不相等的图片，找到最长的一边
    longest_edge = max(h, w)

    #计算短边需要增加多上像素宽度使其与长边等长
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass
    #RGB颜色
    BLACK = [0, 0, 0]
    #给图像增加边界，是图片长、宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value = BLACK)
    #调整图像大小并返回
    return cv2.resize(constant, (height, width))

# http://blog.csdn.net/honghu549599aaa/article/details/51275349
def colorDetect(img,option=0):
    # name = random.randint(0,99)
    img = resize_image(img,900,900)
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #高斯模糊
    img = cv2.GaussianBlur(img,(5,5),0)
    # cv2.imshow("im", img)
    # cv2.waitKey(0)
    # 设定蓝色的阈值
    if(option == 0):
        lower=np.array([100,50,50])
        upper=np.array([140,255,255])
    else:
        #黄色
        lower=np.array([15,50,50])
        upper=np.array([40,255,255])

    # 根据阈值构建掩模
    mask=cv2.inRange(hsv,lower,upper)
    # cv2.imshow("im", mask)
    # cv2.waitKey(0)
    # 对原图像和掩模进行位运算
    res=cv2.bitwise_and(img,img,mask=mask)
    # cv2.imshow("im", res)
    # cv2.waitKey(0)
    gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
    #二值化
    ret,thresh1 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # cv2.imshow("im", thresh1)
    # cv2.waitKey(0)
    #闭操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(17, 3))
    closed = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("im", closed)
    # cv2.waitKey(0)
    contours_img, contours, hierarchy = cv2.findContours(closed.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    imgRs = []
    i = 0
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)

        # box = np.int0(box)
        # cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
        # cv2.waitKey(0)
        if(w<50 or h < 15 or w < h ):
            continue
        if((w / h) < 2.5 or (w / h)> 5):
            continue
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        # print(box)
        # print('++++++=')
        rs = img[y:y+h,x:x+w]
        imgRs.append(rs)
    return imgRs


# if __name__ == '__main__':
#     imagePath = 'aa.jpg' # 图片路径s
#     img = cv2.imread(imagePath)
#     res = colorDetect(img)
#     for re in res:
#         cv2.imshow('pr.jpg', re)
#         #cv2.imshow("img",img)
#         cv2.waitKey(0)