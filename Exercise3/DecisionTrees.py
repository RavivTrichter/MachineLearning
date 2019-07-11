import numpy as np
from sklearn import datasets

# calculates the impurity of a given group Xt
def Impurity(Xt):
    Xt = np.copy(Xt)
    if len(Xt) <= 1:
        return 1
    values, counts = np.unique(Xt[:, -1], return_counts=True)  # the last column has is the y-vector.
    sum_of_groups = np.sum(counts)
    return np.sum([-(group / sum_of_groups) * np.log2(group / sum_of_groups) for group in counts])

# calculates the delta given the three groups Xt, Xt_n, Xt_y
def delta(Xt, Xt_n, Xt_y):
    return Impurity(Xt) - ((len(Xt_y) / len(Xt)) * Impurity(Xt_y)) - (
            (len(Xt_n) / len(Xt)) * Impurity(Xt_n))

# reads the irisdata from sklearn
def readData():
    iris = datasets.load_iris()
    X = iris.data[:, :]
    y = iris.target
    X_y = [[X[i, 0], X[i, 1], X[i, 2], X[i, 3], y[i]] for i in range(len(y))] # appending the y to each vector
    return np.array(X_y), np.array(y)

#  feature_idx is the index which w
def split(Xt, alpha, attribute_idx, stop_split_criterion):
    Xt_n = []
    Xt_y = []
    Iris_dict = dict()
    Iris_dict[0] = "setosa"
    Iris_dict[1] = "versicolor"
    Iris_dict[2] = "virginica"
    if len(Xt) < 1:  # the given node has no data inside
        return list(), list()

    for group in Xt:  # splits the groups into two sub groups Xt_n and Xt_y
        Xt_n.append(group) if group[attribute_idx] <= alpha else Xt_y.append(group)

    diff_group = len(set(Xt[:, -1]))  # returns the different numbers (group ID's) for each group [0, 1 ,2] in irisdata
    if delta(Xt, Xt_n, Xt_y) < stop_split_criterion or diff_group < 2:  # is a leaf

        Xt_n_groups = [group[-1] for group in Xt_n]  # gives us the vector ID of all groups in Xt_n
        Xt_y_groups = [group[-1] for group in Xt_y]  # gives us the vector ID of all groups in Xt_y
        values_n, counts_n = np.unique(Xt_n_groups, return_counts=True)
        values_y, counts_y = np.unique(Xt_y_groups, return_counts=True)
        if len(values_n) < 1 or len(values_y) < 1:
            return [], []  # nothing to split anymore
        if np.max(counts_n) > np.max(counts_y):  # returns a tuple -> we want to know who has the most occurrences
            print("it's a leaf\nXt_n : The most common group is :", Iris_dict[values_n[np.argmax(counts_n)]], "and it's cardinality is :", np.max(counts_n), "\n")
        else:
            print("it's a leaf\nXt_y : The most common group is :", Iris_dict[values_y[np.argmax(counts_y)]], "and it's cardinality is :", np.max(counts_y), "\n")
        return list(), list()  # nothing to split anymore
    return np.copy(Xt_n), np.copy(Xt_y)


def select_feature_threshold(Xt, stop_split_criterion):
    best_impurity = np.inf
    best_alpha = 0.0
    attribute = 0
    best_Xt_n = []
    best_Xt_y = []
    for i in range(len(Xt[0])-1):  # run on all the attributes except the last which is the y-vector
        attribute_i = np.array(Xt[:, i])
        for j in range(len(attribute_i)-1):  # find the best alpha
            alpha = (attribute_i[j] + attribute_i[j+1]) / 2
            Xt_n, Xt_y = split(Xt, alpha, i, stop_split_criterion)
            if len(Xt_n) <= 1 or len(Xt_y) <= 1:
                return best_impurity, best_alpha, attribute, best_Xt_n, best_Xt_y
            #  the row below gives us an indication whether the impurity has decreased after the split
            curr_impurity = Impurity(Xt_n)*(len(Xt_n))/len(Xt) + Impurity(Xt_y)*(len(Xt_y))/len(Xt)
            if curr_impurity < best_impurity:
                best_impurity = curr_impurity
                best_alpha = alpha
                attribute = i
                best_Xt_n = np.copy(Xt_n)
                best_Xt_y = np.copy(Xt_y)
    return best_impurity, best_alpha, attribute, best_Xt_n, best_Xt_y


def createTree(Xt, num_of_groups, stop_split_criterion):
    if len(Xt) < 1:
        return
    best_impurity, best_alpha, attribute, Xt_n, Xt_y = select_feature_threshold(Xt, stop_split_criterion)
    createTree(Xt_n, num_of_groups, stop_split_criterion)
    createTree(Xt_y, num_of_groups, stop_split_criterion)




if __name__ == "__main__":
    X_y, y = readData()
    num_of_groups = len(set(y)) # number of different classes
    createTree(X_y, num_of_groups, stop_split_criterion=0.003)