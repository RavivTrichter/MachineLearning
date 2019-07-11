import numpy as np
import matplotlib.pyplot as plt

x = []
x_log = []
y = []
y_log = []

def read_data(filename):
    file = open(filename, 'r')
    for line in file:
        tmp_x,tmp_y = line.split()
        x.append([1.0, float(tmp_x)])
        x_log.append([1.0, np.log(float(tmp_x))])
        y.append(float(tmp_y))

def cost_function(x, y, m, theta):
    return np.sum(hypotheses(x, theta) - y)**2/(2*m)


def hypotheses(x, theta):
    return np.dot(x, theta)


def Gradient_Descent(x, y, theta, alpha, max_iter=100):
    Jiter = []
    m = len(y)
    i = 0
    while i < max_iter:
        theta = np.copy(theta - alpha*(1/m)*np.dot((hypotheses(x, theta)-y), x))
        Jiter.append(cost_function(x, y, m, theta))
        i = i + 1
    return theta, Jiter


if __name__ == '__main__':
    read_data("moore.dat")
    y_log = np.log(y)
    theta = np.zeros(2)
    alpha = 1e-3
    max_iter = 500
    theta, Jiter = Gradient_Descent(x_log, y_log, theta, alpha, max_iter)
    print(theta)
    plt.plot(Jiter)
    plt.savefig('section5_1.png')
    plt.show()
    num_of_transistors_2014 = 4300000000
    print("Number of Transistors in 2017 : ",(1*theta[0] + num_of_transistors_2014*theta[1]))

