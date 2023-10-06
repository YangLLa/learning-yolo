# ！/user/bin/env python
# -*- coding:UTF-8 -*-
# author: yanglulu time:2023/4/11


#将一张三通道图片转换为RGB三通道灰度图，并将每个通道的矩阵输出
import cv2
import matplotlib.pyplot as plt
'''
彩色图像转化为RGB三幅灰度图像
'''
def main():
  img='./0049.jpg'
  im=cv2.imread(img)
  B,G,R=cv2.split(im)
  cv2.imshow("R",R)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  cv2.imshow("G", G)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  cv2.imshow("B", B)
  cv2.waitKey(0)
  cv2.destroyAllWindows()


  # #结果展示
  # plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文乱码
  # #子图1，原始图像
  # plt.subplot(141)
  # #plt默认使用三通道显示图像，所以需要制定cmap参数为gray
  # #imshow()对图像进行处理，画出图像，show()进行图像显示
  # #opencv的颜色通道顺序为[B,G,R]，而matplotlib颜色通道顺序为[R,G,B],所以需要调换一下通道位置
  # plt.imshow(im[:,:,(2,1,0)])
  # plt.title('原图像')
  # #不显示坐标轴
  # plt.axis('off')
  #
  # #子图2，通道R灰度图像
  # plt.subplot(142)
  # cv2.imshow(R)
  # plt.title('通道R')
  # plt.axis('off')
  # print("channel[R]")
  # print(im[:, :, 2])
  #
  # #子图3，通道G
  # plt.subplot(143)
  # cv2.imshow(G)
  # plt.title('通道G')
  # plt.axis('off')
  # print("channel[G]")
  # print(im[:, :, 1])
  #
  # #子图4，B
  # plt.subplot(144)
  # cv2.imshow(B)
  # plt.title('通道B')
  # plt.axis('off')
  # print("channel[B]")
  # print(im[:, :, 0])
  #
  # plt.show()


if __name__== '__main__':
  main()


# USAGE
# python splitting_and_merging.py --image ../images/wave.png

# Import the necessary packages
import numpy as np
import argparse
import cv2


image = './0049.jpg'
im = cv2.imread(image)


(B, G, R) = cv2.split(im)

# # Show each channel individually
# cv2.imshow("Red", R)
# cv2.imshow("Green", G)
# cv2.imshow("Blue", B)
# cv2.waitKey(0)
#
# # Merge the image back together again
# merged = cv2.merge([B, G, R])
# cv2.imshow("Merged", merged)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Now, let's visualize each channel in color
zeros = np.zeros(im.shape[:2], dtype = "uint8")
cv2.imshow("Red", cv2.merge([zeros, zeros, R]))
cv2.imshow("Green", cv2.merge([zeros, G, zeros]))
cv2.imshow("Blue", cv2.merge([B, zeros, zeros]))
cv2.waitKey(0)
