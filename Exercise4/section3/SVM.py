import numpy as np
import matplotlib.pyplot as plt
import cython
from qpsolvers import solve_qp
from section3 import Perceptron as percept


def svm_train(x, y, d):
    H = np.eye(d)
    f = np.zeros(d)
    Dy = np.diagflat(y)
    A = Dy @ x
    b = np.ones(len(y))
    theta = solve_qp(H, f, -A, -b)
    return theta



def svm_test(theta, x_test, y_test):
    print("test")


def plot(theta, X, y, margin):
    x0 = np.array([X[i] for i in range(len(y)) if y[i] == 1])
    x1 = np.array([X[i] for i in range(len(y)) if y[i] == -1])
    plt.scatter(x1[:, 0], x1[:, 1], marker="D", color="g")
    plt.scatter(x0[:, 0], x0[:, 1], marker="X", color="r")
    pts_x = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 50)
    pts_x1 = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 50)
    pts_x2 = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 50)
    pts_y = np.array([(-theta[0] * i) / theta[1] for i in pts_x])
    pts_y1 = np.array([(-theta[0] * i) / theta[1] + margin for i in pts_x1])
    pts_y2 = np.array([(-theta[0] * i) / theta[1] - margin for i in pts_x2])
    plt.scatter(pts_x, pts_y, color="b")
    plt.scatter(pts_x1, pts_y1)
    plt.scatter(pts_x2, pts_y2)
    plt.scatter(pts_x, pts_y)
    plt.show()


if __name__ == "__main__":
    d = 2
    print("Data1.mat : \n")

    X_a, y_a = percept.readData('data1.mat')
    theta_a = svm_train(X_a, y_a, d)
    theta_perceptron, k = percept.myPerceptronTrain(X_a, y_a)
    margin = percept.calculateGeometricMargin(theta_a, X_a)
    plot(theta_a, X_a, y_a, margin)
    print("SVM theta : ", theta_a, "\nPerceptron theta :", theta_perceptron)
    print("SVM margin : ", margin, "\nPerceptron margin :", percept.calculateGeometricMargin(theta_perceptron, X_a))

    print("\n\nData2.mat :\n")

    X_b, y_b = percept.readData('data2.mat')
    print("SVM test : ",str (percept.myPerceptronTest(theta_a, X_b, y_b) * 100) + "% precision" )
    print("Perceptron test : ", str(percept.myPerceptronTest(theta_perceptron, X_b, y_b) * 100) + "% precision")



