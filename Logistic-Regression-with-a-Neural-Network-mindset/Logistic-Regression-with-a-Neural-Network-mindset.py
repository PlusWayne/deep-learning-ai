import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

# loading the data
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Reshape the traing data and test examples

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# Preprocessing data

train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255


# def sigmoid
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


# initialize with zeros
def initialize_with_zeros(dim):
    w = np.zeros(shape=(dim, 1))
    b = 0
    return w, b


# forward propagate and calculate gradient
def propagate(w, b, X, Y):
    m = X.shape[1]

    # forward propagate
    A = sigmoid(np.dot(w.T, X) + b)
    # loss or cost
    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    # calculate gradient
    dw = 1 / m * np.dot(X, (A - Y).T)
    db = 1 / m * np.sum(A - Y)

    # squeeze to make sure cost is a real number
    cost = np.squeeze(cost)

    grads = {'dw': dw,
             'db': db}
    return grads, cost


# optimize weight and biases
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []
    # train
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        w = w - learning_rate * grads['dw']
        b = b - learning_rate * grads['db']

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {'w': w,
              'b': b}

    grads = {"dw": grads['dw'],
             "db": grads['db']}
    return params, grads, costs


# predict
def predict(w, b, X):
    Y_prediction = np.zeros(shape=(1, X.shape[1]), dtype=int)
    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0
    return Y_prediction


# model
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w, b = initialize_with_zeros(dim=X_train.shape[0])
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = parameters['w']
    b = parameters['b']

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d


d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)

costs = np.squeeze(d['costs'])

plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()


my_image = "mycat.jpg"

# preprocess the image to fit your algorithm.
num_px = 64
fname = my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px, num_px)).reshape((1, num_px * num_px * 3)).T
my_predicted_image = predict(d["w"], d["b"], my_image)

# plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[
    int(np.squeeze(my_predicted_image)),].decode("utf-8") + "\" picture.")
