import numpy as np
import matplotlib.pyplot as plt
import random
import sys
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabaz_score

def data_generation(dimension=10):
    """
    Train data generation
    :param dimension: the dimension of the data(default:10)
    :return: the training data, size is 90*10
    """
    np.random.seed(5)
    data1 = np.random.randint(30, 100, [30, dimension])
    data2 = np.random.randint(0, 70, [30, dimension])
    data3 = np.random.randint(-50, 70, [30, dimension])
    return np.concatenate((data1, data2, data3), axis=0)


def visualization_origin_data(data):
    """
    Visualization of the origin data
    :param data: the original data
    :return: a scatter plot
    """
    data1 = TSNE(n_components=2, init='pca', random_state=89).fit_transform(data)
    plt.scatter(data1[:, 0], data1[:, 1], s=20)
    plt.title("Origin Data Visualization")
    plt.savefig("origin_data.jpg")
    plt.show()


def cal_distance(x, y, mode = "euclidean"):
    """
    Calculate the distance between x and y
    :param x:
    :param y:
    :return:
    """
    if mode == "euclidean":
        return np.sqrt(np.sum((x-y)**2))
    elif mode == "euclideanSquare":
        return np.sum((x-y)**2)
    else:
        return np.sum(x-y)


def my_silhouette_score(data, label, mode):
    """
    Calculate the Silhouette Coefficient(轮廓系数)
    :param data: train data
    :param label: cluster label
    :param mode: distance mode
    :return:
    """
    score = []
    assert len(data)==len(label)
    n = len(data)           # 数据个数
    m = len(set(label))     # 簇的个数

    # 对数据集中的每一个样本计算其s_i，最后求均值
    for i in range(n):
        lable_i = label[i]
        same_list = []
        for j in range(n):
            if (label[j]==lable_i and i!=j): same_list.append(j)

        # 计算a_i
        a_i = []
        for index in same_list:
            a_i.append(cal_distance(data[i, :], data[index, :], mode=mode))
        a_i = np.mean(np.array(a_i))

        # 计算b_i
        b_i = sys.float_info.max
        for other_lable in set(label):
            if other_lable != lable_i:
                other_lable_index_list = []
                for j in range(n):
                    if (label[j] == other_lable): other_lable_index_list.append(j)
                other_lable_distance = []
                for index in other_lable_index_list:
                    other_lable_distance.append(cal_distance(data[i, :], data[index, :], mode=mode))
                other_lable_distance = np.mean(np.array(other_lable_distance))
                b_i = min(b_i, other_lable_distance)
        # 计算s_i
        s_i = (b_i - a_i)/(max(a_i, b_i))
        score.append(s_i)
    return np.mean(np.array(score))


def sk_KMeans_algorithm(data, n_clusters=3):
    """
    调用sklearn自带的KMeans算法进行验证
    :param data:
    :return:
    """
    estimator = KMeans(n_clusters=n_clusters)
    estimator.fit(data)
    label_pred = estimator.labels_
    # centroids = estimator.cluster_centers_
    # inertia = estimator.inertia_
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    j = 0
    data2 = TSNE(n_components=2, init='pca', random_state=89).fit_transform(data)
    for i in label_pred:
        plt.plot([data2[j:j + 1, 0]], [data2[j:j + 1, 1]], mark[i], markersize=5)
        j += 1
    plt.title("sklearn KMeans")
    plt.show()
    sc_score_2 = my_silhouette_score(data, label_pred, mode='euclidean')
    sc_score_3 = silhouette_score(data, label_pred, metric='euclidean')
    ch_score_2 = calinski_harabaz_score(data, label_pred)
    return sc_score_2, sc_score_3, ch_score_2


def my_KMeans_algorithm(data, n_clusters=3):
    """
    the KMeans Algorithm
    :return:
    """
    size = len(data)

    # clusterAssment是一个二维矩阵
    # 第一维记录样本点属于哪个类
    # 第一维记录该样本点到类中心的距离
    clusterAssment = np.array(np.zeros((size, 2)))

    # 初始化质心：随机选取数据集中的n_clusters个点
    centerIndex = random.sample(range(0, size), n_clusters)
    center = data[centerIndex, :]
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']

    change = True
    iter = 0   # 记录迭代次数

    # 只要样本点所属的类还发生变化，就继续迭代
    while change:
        change = False

        # step one: 根据当前质心划分数据集
        for i in range(size):
            minDistance = sys.float_info.max
            minIndex = -1
            for j in range(n_clusters):
                distance = cal_distance(data[i, :], center[j, :], mode="euclideanSquare")
                if distance < minDistance:
                    minDistance = distance
                    minIndex = j

            # 更新样本点所属类以及到质心的距离
            if int(clusterAssment[i][0]) != int(minIndex):
                clusterAssment[i][0] = minIndex
                clusterAssment[i][1] = minDistance
                change = True

        # step two: 根据新的划分计算新的质心
        for j in range(n_clusters):
            index = []
            for i in range(size):
                if(int(clusterAssment[i][0]) == j): index.append(i)
            center[j, :] = np.mean(data[index, :], axis=0)

        iter += 1

        # 可视化
        plot_data = TSNE(n_components=2, init='pca', random_state=89).fit_transform(data)
        j = 0
        for i in clusterAssment:
            i = int(i[0])
            plt.plot([plot_data[j:j + 1, 0]], [plot_data[j:j + 1, 1]], mark[i], markersize=5)
            j += 1
        plt.title("t="+str(iter))
        plt.show()

    # 计算轮廓系数
    label_pred = []
    for item in clusterAssment:
        label_pred.append(int(item[0]))
    sc_score_1 = my_silhouette_score(data, label_pred, mode='euclidean')

    # 计算Calinski-Harabasz系数
    ch_score_1 = calinski_harabaz_score(data, label_pred)
    print("The iteration times is : "+str(iter))
    return sc_score_1, ch_score_1, iter


if __name__ == "__main__":

    n_clusters = 8                    # numbers of the clusters
    data = data_generation()           # data generation
    visualization_origin_data(data)    # visualization

    # 调用sklearn自带的KMeans算法进行验证并可视化结果
    sc_score_2, sc_score_3, ch_score_2 = sk_KMeans_algorithm(data, n_clusters=n_clusters)

    # 多次实验
    sc_score = []
    ch_score = []
    iteration = []
    for _ in range(10):
        sc_score_1, ch_score_1, iter = my_KMeans_algorithm(data=data, n_clusters=n_clusters)
        sc_score.append(sc_score_1)
        ch_score.append(ch_score_1)
        iteration.append(iter)
    print("k = " + str(n_clusters))
    print("times " + str(np.mean(np.array(iteration))))
    print(np.mean(np.array(sc_score)))
    print(sc_score_2)
    print(sc_score_3)
    print(np.mean(np.array(ch_score)))
    print(ch_score_2)
