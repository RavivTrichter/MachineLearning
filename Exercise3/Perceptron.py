import numpy as np
import scipy.io as sio
import math
import matplotlib.pyplot as plt


# Section 1 - the two functions MyPerceptronTrain ^ MyPerceptronTest

def readData(filename):
    data = sio.loadmat(filename)
    X = data['X']
    y = data['y']
    return X, y


# we assume the data is linear seperable, return theta,k
# X as a MxN matrix and y is a Mx1 vector while n is the dimension and m is the number of training vectors
def myPerceptronTrain(X, y):
    m = len(y)
    n = len(X[0])
    theta = np.zeros(n)  # theta is of dimension n
    right_predictions = 0
    k = 0
    while right_predictions < m:
        prediction = theta @ X[k % m] * y[k % m]
        if prediction > 0:  # predicted good
            right_predictions += 1
        else:
            right_predictions = 0
            theta += X[k % m] * y[k % m]  # tuning theta
        k += 1
    return theta, k

# Returns the error percentage
def myPerceptronTest(theta, X_test, y_test):
    correct = [1 if np.sign(theta @ X_test[k]) == y_test[k] else 0 for k in range(len(y_test))]
    return 1 - (np.sum(correct)/ len(correct))  # subtracting from 1 the percent of success -> gives me the error

# calculating the angle between two 2D vectors
def calculateAngle(a, b):
    return math.degrees(np.arccos((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))))

# calculating the distance from a point to (theta^T * x = 0) line
def calculateDistanceToZero(x,theta):
    return np.abs(x @ theta / np.linalg.norm(theta))

# calculates the closest point to theta @ x = 0
def calculateGeometricMargin(theta,X):
    return np.min([calculateDistanceToZero(X[i], theta) for i in range(len(X))])

def calculateMaxNorm(X):
    return np.max([np.linalg.norm(X[i]) for i in range(len(X))])


def plot(X, y, theta, filename):
    x0 = np.array([X[i] for i in np.where(y == 1)])
    x1 = np.array([X[i] for i in np.where(y == -1)])
    plt.scatter(x1[0, :, 0], x1[0, :, 1], marker="D", color="g")
    plt.scatter(x0[0, :, 0], x0[0, :, 1], marker="X", color="r")
    pts_x = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 500)
    pts_y = np.array([(theta[0] * -i) / theta[1] for i in pts_x])
    plt.scatter(pts_x, pts_y)
    plt.savefig(filename)
    plt.show()


if __name__ == "__main__":

    # Section 2.a
    print("Section 2.a")
    X_a, y_a = readData('data1.mat')
    theta_a, k_a = myPerceptronTrain(X_a, y_a)
    print("Theta_a : ", theta_a, "\nThe number of iterations until convergence is : ", k_a)
    print("The number of errors is : ", str(myPerceptronTest(theta_a, X_a, y_a)*100) + "%")
    print("The angle between theta_a and [1,0] is : ", calculateAngle(np.transpose([1, 0]), theta_a))

    # Section 2.b
    print("\n\nSection 2.b")
    X_b, y_b = readData('data2.mat')
    theta_b, k_b = myPerceptronTrain(X_b, y_b)
    print("Theta_b : ", theta_b, "\nThe number of iterations until convergence is : ", k_b)
    print("The number of errors is : ", str(myPerceptronTest(theta_b, X_b, y_b)*100) + "%")
    print("The angle between theta_b and [1,0] is : ", calculateAngle(np.transpose([1, 0]), theta_b))


    # Section 2.c
    print("\n\nSection 2.c")
    print("The Geometric Margin for 'data1.mat' is : ", calculateGeometricMargin(theta_a, X_a))
    print("The Geometric Margin for 'data2.mat' is : ", calculateGeometricMargin(theta_b, X_b))


    # Section 2.d
    print("\n\nSection 2.d")
    print("The maximum norm in 'data1.mat' is : ", calculateMaxNorm(X_a))
    print("The maximum norm in 'data2.mat' is : ", calculateMaxNorm(X_b))


    # Section 2.e
    plot(X_a, y_a, theta_a, 'plot_data_1.png')
    plot(X_b, y_b, theta_b, 'plot_data_2.png') # we can see that it's 'data2.mat' is harder