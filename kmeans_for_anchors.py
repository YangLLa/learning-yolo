# -*- coding: utf-8 -*-
import numpy as np
import glob
def iou(box, clusters):
    """
    计算一个ground truth边界盒和k个先验框(Anchor)的交并比(IOU)值。
    参数box: 元组或者数据，代表ground truth的长宽。
    参数clusters: 形如(k,2)的numpy数组，其中k是聚类Anchor框的个数
    返回：ground truth和每个Anchor框的交并比。
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")
    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]
    iou_ = intersection / (box_area + cluster_area - intersection)
    return iou_

def avg_iou(boxes, clusters):
    """
    计算一个ground truth和k个Anchor的交并比的均值。
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])

def kmeans(boxes, k, dist=np.median):
    """
    利用IOU值进行K-means聚类
    参数boxes: 形状为(r, 2)的ground truth框，其中r是ground truth的个数
    参数k: Anchor的个数
    参数dist: 距离函数
    返回值：形状为(k, 2)的k个Anchor框
    """
    # 即是上面提到的r
    rows = boxes.shape[0]
    # 距离数组，计算每个ground truth和k个Anchor的距离
    distances = np.empty((rows, k))
    # 上一次每个ground truth"距离"最近的Anchor索引
    last_clusters = np.zeros((rows,))
    # 设置随机数种子
    np.random.seed(0)

    # 初始化聚类中心，k个簇，从r个ground truth随机选k个
    clusters = boxes[np.random.choice(rows, k, replace=False)]
    # 开始聚类
    while True:
        # 计算每个ground truth和k个Anchor的距离，用1-IOU(box,anchor)来计算
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)
        # 对每个ground truth，选取距离最小的那个Anchor，并存下索引
        nearest_clusters = np.argmin(distances, axis=1)
        # 如果当前每个ground truth"距离"最近的Anchor索引和上一次一样，聚类结束
        if (last_clusters == nearest_clusters).all():
            break
        # 更新簇中心为簇里面所有的ground truth框的均值
        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)
        # 更新每个ground truth"距离"最近的Anchor索引
        last_clusters = nearest_clusters

    return clusters

# 加载自己的数据集，只需要所有labelimg标注出来的xml文件即可
def load_dataset(PATH):
    dataset = []

    for filename in glob.glob(PATH):
        with open(filename, "r") as f:
            for line in f.readlines():
                if line.strip():
                    line = line.split()
                    w = float(line[3])
                    h = float(line[4])
                    dataset.append([w, h])

    return np.array(dataset)


if __name__ == '__main__':
    #ANNOTATIONS_PATH = "./my_yolo_dataset/all_anchor/*.txt"  # xml文件所在文件夹
    ANNOTATIONS_PATH = r"C:\Users\yanglulu288\Desktop\insulator\Annoation\*.xml"  # xml文件所在文件夹
    CLUSTERS = 9  # 聚类数量，anchor数量
    INPUTSIZE = [512, 512]  # 输入图像大小
    data = load_dataset(ANNOTATIONS_PATH)
    data[:, 0] = data[:, 0] * INPUTSIZE[0]
    data[:, 1] = data[:, 1] * INPUTSIZE[1]
    out = kmeans(data, k=CLUSTERS)
    print('Boxes:')
    print(out)
    print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
    final_anchors = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
    print("Before Sort Ratios:\n {}".format(final_anchors))
    print("After Sort Ratios:\n {}".format(sorted(final_anchors)))
    index = np.argsort(out[:, 0] * out[:, 1])
    print(out[index])