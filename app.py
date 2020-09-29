import math

import pandas as pd
import random
import numpy as np
import collections
import matplotlib.pyplot as plt

MAX_ITERATIONS = 100
centroid_dict = {}
SSE = {}
global finalDF


def q1Kmeans(clusterDF):
    global centroid_dict
    k = 2
    while (k <= 20):
        initializeCentroids(clusterDF, k)  # initialize random Cluster
        assignNearestItem(clusterDF, k)  # assign nearest neighbor
        k = k + 2


def initializeCentroids(clusterDF, k):
    global centroid_dict
    for i in range(1, k + 1):
        coordinateList = []
        for column in clusterDF:
            a = int(clusterDF[column].min())
            b = int(clusterDF[column].max())
            vi = random.randint(a, b)
            coordinateList.append(vi)

        centroid_dict[i] = coordinateList


def find_euclidean_distance(row):
    global centroid_dict
    distance_list = []
    for keyLabel in centroid_dict:
        x = pd.Series(centroid_dict.get(keyLabel))
        y = pd.Series(row)
        dist = np.sqrt(np.sum([(a - b) * (a - b) for a, b in zip(x, y)]))
        distance_list.append(float(dist))

    min_value = min(distance_list)
    clusterLabel = distance_list.index(min_value) + 1
    return clusterLabel


def assignNearestItem(clusterDF, k):
    global MAX_ITERATIONS
    global centroid_dict
    global SSE
    global finalDF
    iteration = 0

    final_label_Cluster_DF = None

    # looping till max iteration is reached
    while iteration < MAX_ITERATIONS:
        label_Cluster_DF = clusterDF.copy()
        label_Cluster_DF['Label'] = 'null'
        # finding distance and assign it to nearest cluster
        for i in range((clusterDF.shape[0])):
            label = find_euclidean_distance(list(clusterDF.iloc[i, :]))
            label_Cluster_DF.iloc[i, label_Cluster_DF.columns.get_loc('Label')] = label

        prev_centroid_dict = centroid_dict.copy()
        for keyLabel in centroid_dict:
            new_centroid_list = []
            for column in clusterDF:
                mean = label_Cluster_DF[label_Cluster_DF['Label'] == keyLabel][column].mean()
                new_centroid_list.append(mean)
            centroid_dict[keyLabel] = new_centroid_list

        isEqual_count = 0
        for keyLabel in centroid_dict:
            checkStagnant = checkEqual(centroid_dict.get(keyLabel), prev_centroid_dict.get(keyLabel))
            if checkStagnant == False:
                break
            else:
                isEqual_count = isEqual_count + 1

        if isEqual_count == len(centroid_dict):
            break
        iteration = iteration + 1
        final_label_Cluster_DF = label_Cluster_DF

    sseValue = 0
    for i in range((final_label_Cluster_DF.shape[0])):
        sseValue = sseValue + calculateSSE(list(final_label_Cluster_DF.iloc[i, :]))

    SSE[k] = sseValue
    print(SSE)
    finalDF = final_label_Cluster_DF

    if k >= 13:
        finalDF.to_csv("EMoptimalK.csv")

    if k == 10:
        finalDF.to_csv("optK10.csv")


def calculateSSE(row):
    global centroid_dict
    cluster_label = row[-1]
    x = pd.Series(row[0: len(row) - 1])
    y = pd.Series(centroid_dict.get(cluster_label))
    dist = np.sum([(a - b) * (a - b) for a, b in zip(x, y)])
    return dist


def checkEqual(centroid_list1, centroidlist2):
    test_list1 = [f"{num:.2f}" for num in centroid_list1]
    test_list2 = [f"{num:.2f}" for num in centroidlist2]
    if collections.Counter(test_list1) == collections.Counter(test_list2):
        return True
    else:
        return False


def plot_elbow_graph():
    plt.plot(pd.Series(list(SSE.keys())), pd.Series(list(SSE.values())), 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('SSE')
    plt.title('SSE VS K (The Elbow Method)')
    plt.show()


if __name__ == "__main__":
    filePath = input("Enter the filepath")
    clusterDF = pd.read_excel(filePath)
    # clusterDF = pd.read_excel("BDAHw/Clustering.xlsx")
    clusterDF.info()
    print(clusterDF.describe())
    print(clusterDF.head())
    q1Kmeans(clusterDF)
    plot_elbow_graph()
