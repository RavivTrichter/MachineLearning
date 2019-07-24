import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import operator as op


def euclidean_distance(instance1, instance2, length=2):
    distance = 0
    for i in range(length):
        distance += ((instance1[i] - instance2[i]) ** 2)
    return distance ** 0.5


def get_sub_matrix(x, start, end):
    result = []
    for i in range(start, end):
        result.append(x[i])
    return np.copy(result)


# reads the irisdata from sklearn
def read_data():
    iris = datasets.load_iris()
    X = iris.data[:, :]
    y = iris.target
    x = [[X[i, 2], X[i, 3], y[i]] for i in range(len(y))]  # taking last columns & appending the y to each vector
    return np.array(x), np.array(y)


def divide_and_plot_data(x):
    training = np.concatenate((get_sub_matrix(x, 0, 35), get_sub_matrix(x, 50, 85), get_sub_matrix(x, 100, 135)))
    x0 = training[training[:, -1] == 0]
    x1 = training[training[:, -1] == 1]
    x2 = training[training[:, -1] == 2]
    plt.scatter(x0[:, 0], x0[:, 1],  marker="o", color="g")
    plt.scatter(x1[:, 0], x1[:, 1],  marker="o", color="r")
    plt.scatter(x2[:, 0], x2[:, 1],  marker="o", color="b")
    plt.show()
    testing = np.concatenate((get_sub_matrix(x, 35, 50), get_sub_matrix(x, 85, 100), get_sub_matrix(x, 135, 150)))
    return training, testing


# 0 - setosa, 1 - versicolor, 2 - virginica
def get_neighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclidean_distance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=op.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return np.array(neighbors)


def Vote(neighbours, num_of_classes):
    result = []
    for i in range(num_of_classes):
        result.append(len(neighbours[neighbours[:, -1] == i]))  # putting together all the vectors that are in same class and taking len
    return np.argmax(np.array(result))


def predict(classified_set, k):
    print("for k = ", k, " the prediction is :", end="")
    correct = [1 if classified_set[k][-1] == classified_set[k][-2] else 0 for k in range(len(classified_set))]
    print(str(np.sum(correct)/ float(len(correct)) * 100) + "%")


if __name__ == '__main__':
    x, y = read_data()  # reading the iris data from sklearn
    training_set, testing_set = divide_and_plot_data(x)
    num_of_classes = len(set(list(x[:, -1])))  # number of different classes


    for k in [2,3,5,7,20]:
        classified = []
        for test_vector in testing_set:
            test_vector_neighbours = get_neighbors(training_set, test_vector, k)
            max_vote = Vote(test_vector_neighbours, num_of_classes)
            test_vector = np.append(test_vector, max_vote)
            classified.append(test_vector)
        predict(classified, k)