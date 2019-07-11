import numpy as np
import matplotlib.pyplot as plt

temperatures = []
seconds = []


def read_data(filename):
    data_file = open(filename, 'r')
    for line in data_file:
        nums = line.split()
        temperatures.append([1.0, float(nums[1])]) # for x0 = 1 in every dot product
        seconds.append(float(nums[0]))


def draw_data():
    temp_values = []
    for t in temperatures:
        temp_values.append(t[1])
    plt.plot(temp_values, seconds,'ro')
    plt.ylabel('Chirps/Seconds')
    plt.xlabel('Temperature')
    plt.savefig('1.png')
    plt.show()

def cost_computation(x, y, m, theta):
    return np.sum(hypotheses(x,theta) - y)**2/(2*m)


def hypotheses(x,theta):
    return np.dot(x, theta)


def Gradient_Descent(x, y, theta, alpha, max_iter=100 ):
    Jiter = []
    m = len(y)
    i = 0
    while i < max_iter:
        theta = np.copy(theta - alpha*(1/m)*np.dot((hypotheses(x, theta)-y), x))
        Jiter.append(cost_computation(x, y, m, theta))
        i = i + 1
    return theta, Jiter


if __name__ == "__main__":
    read_data("Xcricket.dat")
    draw_data() # section 1
    theta = np.zeros(2)
    alpha = 2e-5
    max_iters = 100
    theta, Jiter = Gradient_Descent(temperatures, seconds, theta, alpha, max_iters)
    print("Theta values : ", theta)
    plt.ylabel('Cost')
    plt.xlabel('Iteration')
    plt.plot(Jiter)
    plt.savefig('2.png')
    plt.show()
    print("for 90 degrees : ", theta[0]*1 + theta[1]*90) # 18.786827043145337
    print("for 70 degrees : ", theta[0]*1 + theta[1]*70) # 14.612558355378917