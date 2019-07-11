import numpy as np
import matplotlib.pyplot as plt
import random
from section5 import KNN as knn


def read_data(filename):
    x = []
    file = open(filename, 'r')
    for line in file:
        tmp_x, tmp_y = line.split()
        x.append([float(tmp_x), float(tmp_y)])
    return np.array(x)


def draw_data(x):
    plt.scatter(x[:, 0], x[:, 1])
    plt.plot(x[:, 0], x[:, 1], 'ro')
    plt.show()

def plot_cluster(x, centroids):
    x1 = x[x[:, -1] == 0]
    x2 = x[x[:, -1] == 1]
    plt.scatter(x1[:, 0], x1[:, 1],  marker="o", color="g")
    plt.scatter(x2[:, 0], x2[:, 1],  marker="o", color="b")
    for centroid in centroids:
        plt.scatter(centroid[0], centroid[1], marker="D", color='r')
    plt.show()

def cluster(x, k):
    randoms = random.sample(range(len(x)), k)  # returns k random numbers => initializing the centroids randomly
    centroids = []
    for i in range(len(randoms)):
        centroids.append(x[randoms[i]])  # appending k random points in x to centroids
    # centroid initialization step - every point is initialized to a centroid
    min_distance = np.inf
    idx = -1
    new_x = []
    for i in range(len(x)):
        for j in range(len(centroids)):
            distance = knn.euclidean_distance(x[i], centroids[j])
            if distance < min_distance:
                min_distance = distance
                idx = j
        new_x.append(np.append(x[i], idx))  # appending to each point the closest random centroid
        min_distance = np.inf
        idx = -1

    # now every point has a centroid
    x = np.array(new_x)
    centroids = np.array(centroids)
    plot_cluster(x, centroids)  # plotting the first


    #  now we run until convergence and we calculate the mean for every vector according to his centroid
    #  after that we run in another for loop to update the centroids - once there is no chane we stop
    converged = False
    while not converged:
        for i in range(len(x)):
            for j in range(len(centroids)):
                distance = knn.euclidean_distance(x[i], centroids[j])
                if distance < min_distance:
                    min_distance = distance
                    idx = j
            x[i][-1] = idx  # assigning the point x[i] to cluster[j]
            min_distance = np.inf
        centroids_before = np.copy(centroids)
        for i in range(len(centroids)):
            x1 = x[x[:, -1] == i]
            centroids[i] = np.array([np.mean(x1[:, 0]), np.mean(x1[:, 1])])
        if np.array_equal(centroids, centroids_before):  # returns True if they are equal
            converged = True
    plot_cluster(x, centroids)



if __name__ == '__main__':
    x = read_data('faithful.txt')
    draw_data(x)  # part I - plotting the data

    # In cluster I plotted first the data with random centroids and at the end after convergence
    cluster(x, k=2)