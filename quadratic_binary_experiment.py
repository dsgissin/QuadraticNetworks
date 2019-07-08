import matplotlib.pyplot as plt
from keras.datasets import mnist
from quadratic_models import *

# define the parameters of the experiment:
r = 25
lr = 0.1
d = 784
batch_size = 128
iterations = 100


# load and normalize the dataset:
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float')
x_test = x_test.astype('float')
x_train = x_train.reshape((-1, 28*28))
x_test = x_test.reshape((-1, 28*28))
x_train /= 255.
x_test /= 255.

# binarize the dataset:
dig1 = 3
dig2 = 5
x_train = x_train.reshape((-1, 28*28))
x_test = x_test.reshape((-1, 28*28))
x_train = x_train[np.logical_or(y_train == dig1, y_train == dig2)]
y_train = y_train[np.logical_or(y_train == dig1, y_train == dig2)]
y_train[y_train == dig1] = 0
y_train[y_train == dig2] = 1
x_test = x_test[np.logical_or(y_test == dig1, y_test == dig2)]
y_test = y_test[np.logical_or(y_test == dig1, y_test == dig2)]
y_test[y_test == dig1] = 0
y_test[y_test == dig2] = 1


# define the models:
projections_model = binary_squared_model(d=d, r=r, lr=lr)
sgd_model = binary_SGD_squared_model(d=d, r=r, lr=lr)


# train the two models:
train_acc_1 = []
test_acc_1 = []
train_acc_2 = []
test_acc_2 = []
norms_1 = []
norms_2 = []
for i in range(iterations):
    train_acc_1.append(projections_model.evaluate(x_train,y_train))
    train_acc_2.append(sgd_model.evaluate(x_train,y_train, verbose=0)[1])
    test_acc_1.append(projections_model.evaluate(x_test,y_test))
    test_acc_2.append(sgd_model.evaluate(x_test,y_test, verbose=0)[1])
    norms_1.append(projections_model.get_norm())
    norms_2.append(get_binary_frob_norm(sgd_model))
    print("Iteration {e}, Train {a}, Test {b}".format(e=i, a=train_acc_1[-1], b=test_acc_1[-1]))
    print("Iteration {e}, Train {a}, Test {b}".format(e=i, a=train_acc_2[-1], b=test_acc_2[-1]))
    batch = np.random.choice(x_train.shape[0], batch_size, replace=False)
    projections_model.train_on_batch(x_train[batch],y_train[batch])
    sgd_model.train_on_batch(x_train[batch],y_train[batch])

acc1 = projections_model.evaluate(x_test,y_test)
print("Projected Final Accuracy {a}".format(a=acc1))
acc2 = sgd_model.evaluate(x_test,y_test,verbose=0)[1]
print("SGD Final Accuracy {a}".format(a=acc2))

plt.figure()
plt.subplot(211)
plt.plot(np.arange(iterations)+1, test_acc_1)
plt.plot(np.arange(iterations)+1, test_acc_2)
plt.legend(['Projections', 'SGD'])
plt.ylabel('Test Accuracy')
plt.title("Rank {r} Quadratic Model - MNIST 3 vs 5".format(r=r))
plt.subplot(212)
plt.plot(np.arange(iterations)+1, norms_1)
plt.plot(np.arange(iterations)+1, norms_2)
plt.ylabel('Model\'s Frobenius Norm')
plt.xlabel('Iterations')
plt.show()

