#!/usr/bin/env python
# -*- coding: utf8 -*-

import sys
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

XML_EXT = '.xml'
ENCODE_METHOD = 'utf-8'


# pascalVocReader readers the voc xml files parse it
class PascalVocReader:
    """
    this class will be used to get transfered width and height from voc xml files
    """

    def __init__ (self, filepath, width, height):
        # shapes type:
        # [labbel, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)], color, color, difficult]
        self.shapes = []
        self.filepath = filepath
        self.verified = False
        self.width = width
        self.height = height

        try:
            self.parseXML()
        except:
            pass

    def getShapes (self):
        return self.shapes

    def addShape (self, bndbox, width, height):
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        width_trans = (xmax - xmin) / width * self.width
        height_trans = (ymax - ymin) / height * self.height
        points = [width_trans, height_trans]
        self.shapes.append((points))

    def parseXML (self):
        assert self.filepath.endswith(XML_EXT), "Unsupport file format"
        parser = etree.XMLParser(encoding=ENCODE_METHOD)
        xmltree = ElementTree.parse(self.filepath, parser=parser).getroot()
        pic_size = xmltree.find('size')
        size = (int(pic_size.find('width').text), int(pic_size.find('height').text))
        for object_iter in xmltree.findall('object'):
            bndbox = object_iter.find("bndbox")
            self.addShape(bndbox, *size)
        return True


class create_w_h_txt:
    def __init__ (self, vocxml_path, txt_path):
        self.voc_path = vocxml_path
        self.txt_path = txt_path

    def _gether_w_h (self):
        pass

    def _write_to_txt (self):
        pass

    def process_file (self):
        file_w = open(self.txt_path, 'w')
        # print (self.txt_path)
        for file in os.listdir(self.voc_path):
            file_path = os.path.join(self.voc_path, file)
            xml_parse = PascalVocReader(file_path, 512, 512)  # 设置图片归一化大小
            data = xml_parse.getShapes()
            for w, h in data:
                txtstr = str(w) + ' ' + str(h) + '\n'
                # print (txtstr)
                file_w.write(txtstr)
        file_w.close()


class kMean_parse:
    def __init__ (self, path_txt):
        self.path = path_txt
        '''
        n_clusters:簇的个数，即你想聚成几类,即设置k值
        init: 初始簇中心的获取方法
        n_init: 获取初始簇中心的更迭次数，默认会初始10次质心，然后返回最好的结果。
        max_iter: 最大迭代次数
        tol: 容忍度，即kmeans运行准则收敛的条件
        random_state: 随机生成簇中心的状态条件
        '''
        self.km = KMeans(n_clusters=9, init="k-means++", n_init=10, max_iter=500, tol=1e-4,
                         random_state=0)  # 更改n_clusters类别数
        self._load_data()

    def _load_data (self):
        self.data = np.loadtxt(self.path)

    def parse_data (self):
        self.y_k = self.km.fit_predict(self.data)
        print(self.km.cluster_centers_)

    def plot_data (self):
        plt.scatter(self.data[self.y_k == 0, 0], self.data[self.y_k == 0, 1], s=15, c="orange", marker="o")
        plt.scatter(self.data[self.y_k == 1, 0], self.data[self.y_k == 1, 1], s=15, c="green", marker="o")
        plt.scatter(self.data[self.y_k == 2, 0], self.data[self.y_k == 2, 1], s=15, c="blue", marker="o")
        plt.scatter(self.data[self.y_k == 3, 0], self.data[self.y_k == 3, 1], s=15, c="red", marker="o")
        plt.scatter(self.data[self.y_k == 4, 0], self.data[self.y_k == 4, 1], s=15, c="yellow", marker="o")
        plt.scatter(self.data[self.y_k == 5, 0], self.data[self.y_k == 5, 1], s=15, c="black", marker="o")
        plt.scatter(self.data[self.y_k == 6, 0], self.data[self.y_k == 6, 1], s=15, c="gray", marker="o")
        plt.scatter(self.data[self.y_k == 7, 0], self.data[self.y_k == 7, 1], s=15, c="pink", marker="o")
        plt.scatter(self.data[self.y_k == 8, 0], self.data[self.y_k == 8, 1], s=15, c="purple", marker="o")

        # draw the centers
        plt.scatter(self.km.cluster_centers_[:, 0], self.km.cluster_centers_[:, 1], s=50, marker="*",
                    c="gold")  # 五角星大小颜色设置
        plt.legend()
        plt.grid()
        plt.show()
#C:\Users\yanglulu288\Desktop\all_dataset\os_images\annotations

if __name__ == '__main__':
    #whtxt = create_w_h_txt("./my_yolo_dataset/all_anchor", "./data1.txt")  # 指定为voc中xml文件夹路径；data1.txt保存迭代过程点集
    whtxt = create_w_h_txt(r"C:\Users\yanglulu288\Desktop\insulator\Annoation",
                           r"C:\Users\yanglulu288\Desktop\insulator\anchor.txt")
    whtxt.process_file()
    kmean_parse = kMean_parse(r"C:\Users\yanglulu288\Desktop\insulator\anchor.txt")  # 路径和生成文件相同。
    kmean_parse.parse_data()
    kmean_parse.plot_data()