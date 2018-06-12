import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def euclidean_distance(x,y):
    if len(x) == len(y):
        return np.sqrt(np.sum(np.power((x-y),2)))
    else:
        print("Input should be of equal length")
    return None

def lrNorm_distance(x,y,power):
    if len(x) == len(y):
        return np.power(np.sum (np.power(np.abs(x-y),power)),(1/(1.0*power)))
    else:
        print("Input should be of equal length")
    return None

def cosine_distance(x,y):
    if len(x) == len(y):
        return np.dot(x,y) / np.sqrt(np.dot(x,x) * np.dot(y,y))
    else:
        print("Input should be of equal length")
    return None

def jaccard_distance(x,y):
    set_x = set(x)
    set_y = set(y)
    return 1 - len(set_x.intersection(set_y)) / len(set_x.union(set_y))

def hamming_distance(x,y):
    diff = 0
    if len(x) == len(y):
        for char1,char2 in zip(x,y):
            if char1 != char2:
                diff+=1
        return diff
    else:
        print("Input should be of equal length")
    return None
def mapping_function(x):
    output_list = []
    for i in range(len(x)):
        output_list.append(x[i]*x[i])

    output_list.append(x[0]*x[1])
    output_list.append(x[0]*x[2])
    output_list.append(x[1]*x[0])
    output_list.append(x[1]*x[2])
    output_list.append(x[2]*x[1])
    output_list.append(x[2]*x[0])
    return np.array(output_list)
def get_random_data():
    x_1 = np.random.normal(loc = 0.2, scale = 0.2, size = (100,100))
    x_2 = np.random.normal(loc = 0.9, scale = 0.1, size = (100,100))
    x = np.r_[x_1,x_2]
    return x
def form_clusters(x,k):
    no_clusters = k
    model = KMeans(n_clusters = no_clusters,init='random')
    model.fit(x)
    labels = model.labels_
    print(labels)
    sh_score = silhouette_score(x,labels)
    return sh_score


if __name__ == "__main__":
    # 应用映射函数。
    x = np.array([10, 20, 30])
    y = np.array([8, 9, 10])
    tranf_x = mapping_function(x)
    tranf_y = mapping_function(y)
    # 打印输出结果
    print(tranf_x)
    print(np.dot(tranf_x, tranf_y))

    # 打印输出等价于核函数的转换输出结果
    output = np.power((np.dot(x, y)), 2)
    print(output)
    x = get_random_data()
    plt.cla()
    plt.figure(1)
    plt.title("Generated Data")
    plt.scatter(x[:,0],x[:,1])
    plt.show()
    sh_scores = []
    for i in range(1,5):
        sh_score = form_clusters(x,i+1)
        sh_scores.append(sh_score)

    no_clusters = [i+1 for i in range(1,5)]
    plt.figure(2)
    plt.plot(no_clusters,sh_scores)
    plt.title("Cluster Quality")
    plt.xlabel("No of clusters k")
    plt.ylabel("Silhouette Coefficient")
    plt.show()