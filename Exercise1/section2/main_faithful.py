import numpy as np
import matplotlib.pyplot as plt


x = []
y = []

def read_data(filename):
    file = open(filename,'r')
    for line in file:
        tmp_x,tmp_y = line.split()
        x.append([1.0, float(tmp_x)])
        y.append(float(tmp_y))


def draw_data():
    x_vals = []
    for i in x:
        x_vals.append(i[1])
    plt.scatter(x_vals, y)
    plt.plot(x_vals,y, 'ro')
    plt.savefig('section4_1.png')
    plt.show()

def cost_function(x, y, m, theta):
    return np.sum(hypotheses(x,theta) - y)**2/(2*m)


def hypotheses(x,theta):
    return np.dot(x, theta)


def Gradient_Descent(x, y, theta, alpha, max_iter=100 ):
    Jiter = []
    m = len(y)
    i = 0
    while i < max_iter:
        theta = np.copy(theta - alpha*(1/m)*np.dot((hypotheses(x, theta)-y), x))
        Jiter.append(cost_function(x, y, m, theta))
        i = i + 1
    return theta, Jiter



if __name__ == "__main__":
    read_data('faithful.txt')
    draw_data()
    theta = np.zeros(2)
    alpha = 0.01 # Different Alphas (From Yizhar's Lecture) : 1e-3, 1e-2, 5e-3
    max_iter = 2000
    theta, Jiter = Gradient_Descent(x, y, theta, alpha, 2000)
    plt.plot(Jiter)
    # plt.axis([0,200,0,2000])
    plt.savefig('section2_2.png')
    plt.show()
    print("Theta : ", theta)
    print("Next eruction if current eruction duration is 1.5 minutes is : ",theta[0]*1 + theta[1]*1.5)
    print("Next eruction if current eruction duration is 3 minutes is : ",theta[0]*1 + theta[1]*3)
    print("Next eruction if current eruction duration is 5 minutes is : ",theta[0]*1 + theta[1]*5)

