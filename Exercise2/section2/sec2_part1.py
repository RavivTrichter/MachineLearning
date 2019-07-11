import numpy as np
import matplotlib.pyplot as plt

x = []
y = []

def Gradient_Descent(x, y, theta, alpha, max_iter=100):
    Jiter = []
    m = len(y)
    i = 0
    while i < max_iter:
        theta = np.copy(theta - alpha*(1/m)*np.dot((sigmoid(x, theta)-y), x))
        Jiter.append(cost_function(x, y, m, theta))
        i = i + 1
    return theta, Jiter

def cost_function(x,y,m,theta):
    return -(1/m)*np.sum(y*np.log(sigmoid(theta, x)) + (1-y)*np.log(1-sigmoid(theta, x)))


def sigmoid(theta,x):
    return 1/(1 + np.exp(-(np.dot(x, np.transpose(theta)))))


def read_data(filename):

    file = open(filename,'r')
    for line in file:
        values = line.split(",")
        x.append([1.0, float(values[0]), float(values[1])])
        y.append(int(values[2]))


def draw_data(theta=None):
     spam = [i for i in range(len(y)) if y[i] == 0]
     letters = [i for i in range(len(y)) if y[i] == 1]
     green_pts = []
     for i in range(len(spam)):
         tmp = x[spam[i]]
         green_pts.append([tmp[1], tmp[2]])
     red_pts = []
     for i in range(len(letters)):
         tmp = x[letters[i]]
         red_pts.append([tmp[1], tmp[2]])
     red_points = np.array(red_pts)
     green_points = np.array(green_pts)
     plt.scatter(green_points[:,0], green_points[:,1],marker='o',color='g')
     plt.scatter(red_points[:,0],red_points[:,1], marker='D',color='r')
     if theta is not None: # drawing the line to seperade the data
         pts_x = np.linspace(0,2,100)
         pts_y = np.array([-(theta[0] + theta[1]*i)/theta[2] for i in pts_x])
         plt.scatter(pts_x,pts_y)
         plt.savefig('sec2_b.png')
     else:
        plt.savefig('sec2_a.png')
     plt.show()




if __name__ == "__main__":
    # section a
    read_data('email_data.txt')
    draw_data() # first figures
    #section b
    x = np.array(x)
    y = np.array(y)
    theta = np.zeros(len(x[0]))
    alpha = 0.1
    max_iters = 1000
    theta, Jiter = Gradient_Descent(x, y, theta, alpha, max_iters)
    print(theta)
    draw_data(theta)




