import numpy as np
import matplotlib.pyplot as plt

x = []
x_log = []
y = []
y_log = []

def read_data(filename):
    file = open(filename,'r')
    for line in file:
        tmp_x,tmp_y = line.split(",")
        x.append([1.0, float(tmp_x)])
        x_log.append([1.0, np.log(float(tmp_x))])
        y.append(float(tmp_y))

def draw_data():
    x_vals = []
    for i in x_log:
        x_vals.append(i[1])
    plt.plot(x_vals, y_log,'ro')
    plt.savefig('section4_1.png')
    plt.show()

def cost_function(x, y, m, theta):
    return np.sum(hypotheses(x, theta) - y)**2/(2*m)


def hypotheses(x,theta):
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
    read_data("kleibers_law_data.txt")
    y_log = np.log(y)
    draw_data()
    theta = np.zeros(2)
    alpha = 1e-2
    max_iter = 3000
    theta, Jiter = Gradient_Descent(x_log, y_log, theta, alpha, max_iter)
    plt.plot(Jiter)
    plt.savefig('section4_2.png')
    plt.show()
    print("Theta :  ", theta)
    print("predicted number of calories for a 10 kg mammal:", np.exp(1*theta[0] + theta[1]*(np.log(10)/4.18)))
    print("the weight of a mammal that consumes 1.63 kJoul a day:",np.exp((np.log(1.63) - theta[0])/theta[1]))
