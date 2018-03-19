import cv2
import rotate_test as rt

# 1、读取图像，并把图像转换为灰度图像并显示
img = cv2.imread("111.png")  # 读取图片

max_per = 0.95
min_per = 0.05

# img_c = rt.rotate(img,-4)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换了灰度化
gaussian = cv2.GaussianBlur(img_gray, (3, 3), 0, 0, cv2.BORDER_DEFAULT)

# 2、将灰度图像二值化，设定阈值是100
img_thre = gaussian
cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY, img_thre)


# 3、保存黑白图片
cv2.imwrite('thre_res.png', img_thre)

# 4、分割字符
white = []  # 记录每一列的白色像素总和
black = []  # ..........黑色.......
height = img_thre.shape[0]
width = img_thre.shape[1]
white_max = 0
black_max = 0
# 计算每一列的黑白色像素总和
for i in range(width):
    s = 0  # 这一列白色总数
    t = 0  # 这一列黑色总数
    for j in range(height):
        if img_thre[j][i] == 255:
            s += 1
        if img_thre[j][i] == 0:
            t += 1
    white_max = max(white_max, s)
    black_max = max(black_max, t)
    white.append(s)
    black.append(t)
    # print(s)
    # print(t)

arg = True  # False表示白底黑字；True表示黑底白字
# if black_max > white_max:
#     arg = True

# 分割图像
def find_end(start_):
    end_ = start_ + 1
    for m in range(start_ + 1, width - 1):
        if (black[m] if arg else white[m]) > (
                max_per * black_max if arg else max_per * white_max):  # 0.95这个参数请多调整，对应下面的0.05
            end_ = m
            break
    return end_


n = 1
start = 1
end = 2
i =1
while n < width - 2:
    n += 1
    if (white[n] if arg else black[n]) > (min_per * white_max if arg else min_per * black_max):
        # 上面这些判断用来辨别是白底黑字还是黑底白字
        # 0.05这个参数请多调整，对应上面的0.95
        start = n
        end = find_end(start)
        n = end

        print(end)
        if end - start > 5:
            
            cj = img_thre[1:height, start:end]
            cv2.imshow('caijian', cj)
            cv2.imwrite(str(i)+'.png', cj)
            cv2.waitKey(0)
            i = i+1
            
