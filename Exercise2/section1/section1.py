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


def read_data(filename_x, filename_y, flag=False):

    x_file = open(filename_x,'r')
    for line in x_file:
        tmp_x_values = line.split()
        x.append([1.0, float(tmp_x_values[0]), float(tmp_x_values[1])])
    y_file = open(filename_y, 'r')
    for bit in y_file:
        y.append(int(bit))
    if flag:
        return np.array(x), np.array(y)


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
         plt.savefig('sec1_classifying.png')
     else:
        plt.savefig('sec_1.png')
     plt.show()

def compute_predictions(file_x, file_y,theta):
    x,y = read_data(file_x, file_y,True)
    res = []
    for i in range(len(y)):
        if sigmoid(theta,x[i]) >= 0.5:
            res.append(1)
        else:
            res.append(0)
    counter = 0
    for i in range(len(y)):
        if y[i] == res[i]:
            counter += 1
    return counter, len(y)

if __name__ == "__main__":
    read_data('X_email_data1.txt', 'y_email_data1.txt')
    draw_data() # first figure
    x = np.array(x)
    y = np.array(y)
    theta = np.zeros(len(x[0]))
    alpha = 1e-2
    max_iters = 10000
    theta, Jiter = Gradient_Descent(x,y,theta,alpha,max_iters)
    print(theta)
    plt.plot(Jiter)
    plt.show()
    draw_data(theta) # with linear seperable line
    x = []
    y = []
    predicted, sz = compute_predictions('email_test_data_X_test.txt', 'email_test_data_y_test.txt', theta)
    print("number of right predictions : ", predicted, "out of", sz, " ===> ", str(predicted / sz * 100) + "%", "\n")



