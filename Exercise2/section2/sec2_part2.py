import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

x = []
y = []

def Gradient_Descent_Reg(x, y, theta, alpha, lambda_t, max_iter=500):
    Jiter = []
    m = len(y)
    i = 0
    while i < max_iter:
        theta = np.copy(theta - alpha*(1/m)*np.dot((sigmoid(x, theta)-y), x))
        for j in range(1,len(theta)): # regularization update
            theta[j] += (lambda_t/m)*theta[j]
        Jiter.append(cost_function_reg(theta,x, y,lambda_t))
        i = i + 1
    return theta, Jiter

def cost_function_reg(theta,x,y, lambda_t):
    m = len(y)
    cost_curr =  -(1/m)*np.sum(y*np.log(sigmoid(theta, x)) + (1-y)*np.log(1-sigmoid(theta, x)))
    reg_sum = 0
    for i in range(len(theta)): # Regularization update
        reg_sum += theta[i]*theta[i]
    return cost_curr + (lambda_t/(2*m))*reg_sum

def cost_function(theta,x,y, lambda_t):
    m = len(y)
    cost_curr = -(1/m)*np.sum(y*np.log(sigmoid(theta, x)) + (1-y)*np.log(1-sigmoid(theta, x)))
    reg_sum = 0
    for i in range(len(theta)): # Regularization update
        reg_sum += theta[i]*theta[i]
    return cost_curr + (lambda_t/(2*m))*reg_sum, theta


def sigmoid(theta,x):
    return 1/(1 + np.exp(-(np.dot(x, np.transpose(theta)))))


def read_data(filename):

    file = open(filename,'r')
    for line in file:
        values = line.split(",")
        x.append([1.0, float(values[0]), float(values[1])])
        y.append(int(values[2]))


def draw_data(theta=None, return_flag=False):
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
         pts_x = np.linspace(0, 2, 100)
         pts_y = np.array([-(theta[0] + theta[1]*i)/theta[2] for i in pts_x])
         plt.scatter(pts_x,pts_y)
     plt.show()
     if return_flag:
         return green_points, red_points



def map_feature(x1, x2):
    '''
    Maps the two input features to quadratic features.
    Returns a new feature array with more features, comprising of
    X1, X2, X1 ** 2, X2 ** 2, X1*X2, X1*X2 ** 2, etc...
    Inputs X1, X2 must be the same size
    '''
    x1.shape = (x1.size, 1)
    x2.shape = (x2.size, 1)
    degree = 6
    out = np.ones(shape=(x1[:, 0].size, 1))

    for i in range(1, degree + 1):
        for j in range(i + 1):
            r = (x1 ** (i - j)) * (x2 ** j)
            out = np.append(out, r, axis=1)

    return out


def plotDecisionBoundary(theta,green_points, red_points, lambda_t):
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    z = np.zeros(shape=(len(u), len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            z[i, j] = (map_feature(np.array(u[i]), np.array(v[j])).dot(np.array(theta)))

    z = z.T
    plt.contour(u, v, z)
    plt.title('lambda = %f' % lambda_t)
    plt.scatter(green_points[:, 0], green_points[:, 1], marker='o', color='g')
    plt.scatter(red_points[:, 0], red_points[:, 1], marker='D', color='r')
    plt.legend(['y = 1', 'y = 0', 'Decision boundary'])
    plt.savefig('sec2_d_lambda=0.5.png')
    plt.show()




def read_data_2_files(filename_x, filename_y, flag=False):

    x_file = open(filename_x,'r')
    for line in x_file:
        tmp_x_values = line.split()
        x.append([1.0, float(tmp_x_values[0]), float(tmp_x_values[1])])
    y_file = open(filename_y, 'r')
    for bit in y_file:
        y.append(int(bit))
    if flag:
        return np.array(x), np.array(y)



def compute_predictions(file_x, file_y,theta):
    x, y = read_data_2_files(file_x, file_y, True)
    x = map_feature(x[:, 1], x[:, 2])
    res = [1 if sigmoid(theta, x[i]) >= 0.5 else 0 for i in range(len(y))]
    correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(res, y)]
    return np.sum(correct), len(y)





if __name__ == "__main__":
    # section a

    read_data('email_data.txt')
    green_points, red_points = draw_data(return_flag=True) # first figures

    #section b

    X = np.array(x)
    y = np.array(y)

    #section c

    X_feature = map_feature(X[:,1], X[:,2])
    theta = np.zeros(len(X_feature[0]))
    alpha = 1.5
    lambda_t = 0.5 # 0.5,0,1,5,10  ==>  section f
    max_iter = 500
    theta, Jiter = Gradient_Descent_Reg(X_feature, y, theta, alpha, lambda_t, max_iter)
    print("theta:", theta,"\n")

    #section d

    plotDecisionBoundary(theta, green_points, red_points, lambda_t)

    # section e
    y_last = np.copy(y)
    x = []
    y = []
    cnt, size = compute_predictions('X_email_data3.txt', 'y_email_data3.txt', theta)
    print("Predicted : ", cnt, "out of :", size, " ==> ",str(cnt) + "%","\n"*2)

    # section g


    result = opt.fmin_tnc(func=cost_function, x0 = theta, args=(X_feature, y_last, lambda_t))
    print('Thetas found by fmin_tnc function (x0 is initialized to theta): ', result[0], "\n\n")
    optimized_theta = np.copy(result[0])
    x = []
    y = []
    cnt, size = compute_predictions('X_email_data3.txt', 'y_email_data3.txt', optimized_theta)
    print("For optimized theta:\nPredicted : ", cnt, "out of :", size, " ==> ", str(cnt) + "%", "\n")

    plotDecisionBoundary(optimized_theta, green_points, red_points, lambda_t)




    result = opt.fmin_tnc(func=cost_function, x0 = np.zeros(len(theta)), args=(X_feature, y_last, lambda_t))
    print('Thetas found by fmin_tnc function (x0 is initialized to zeros): ', result[0], "\n")
    optimized_theta = np.copy(result[0])
    x = []
    y = []
    cnt, size = compute_predictions('X_email_data3.txt', 'y_email_data3.txt', optimized_theta)
    print("For optimized theta:\nPredicted : ", cnt, "out of :", size, " ==> ", str(cnt) + "%", "\n")

    plotDecisionBoundary(optimized_theta, green_points, red_points, lambda_t)




    result = opt.fmin_tnc(func=cost_function, x0 = np.random.normal(1.0, 0.005, len(theta)), args=(X_feature, y_last, lambda_t))
    print('Thetas found by fmin_tnc function (x0 is initialized to a random vector normaly distributed): ', result[0], "\n")
    optimized_theta = np.copy(result[0])
    x = []
    y = []
    cnt, size = compute_predictions('X_email_data3.txt', 'y_email_data3.txt', optimized_theta)
    print("For optimized theta:\nPredicted : ", cnt, "out of :", size, " ==> ", str(cnt) + "%", "\n")

    plotDecisionBoundary(optimized_theta, green_points, red_points, lambda_t)






