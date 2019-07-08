import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
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
x_test /= np.max(x_train)
x_train /= np.max(x_train)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# define the models:
glram_model = GLRAM_quadratic_model(d=d, r=r, lr=lr)
sgd_model = SGD_quadratic_model(d=d, r=r, lr=lr)


# train the two models:
train_acc_1 = []
test_acc_1 = []
train_acc_2 = []
test_acc_2 = []
for i in range(iterations):
    test_acc_1.append(glram_model.evaluate(x_test,y_test))
    test_acc_2.append(sgd_model.evaluate(x_test,y_test, verbose=0)[1])
    print("Iteration {e}, Test {b}".format(e=i, b=test_acc_1[-1]))
    print("Iteration {e}, Test {b}".format(e=i, b=test_acc_2[-1]))
    batch = np.random.choice(x_train.shape[0], batch_size, replace=False)
    glram_model.train_on_batch(x_train[batch],y_train[batch])
    sgd_model.train_on_batch(x_train[batch],y_train[batch])

acc1 = glram_model.evaluate(x_test,y_test)
print("Projected Final Accuracy {a}".format(a=acc1))
acc2 = sgd_model.evaluate(x_test,y_test,verbose=0)[1]
print("SGD Final Accuracy {a}".format(a=acc2))

plt.figure()
plt.plot(np.arange(iterations)+1, test_acc_1)
plt.plot(np.arange(iterations)+1, test_acc_2)
plt.legend(['GLRAM', 'SGD'])
plt.ylabel('Test Accuracy')
plt.title("Rank {r} Quadratic Model - Multiclass MNIST".format(r=r))
plt.xlabel('Iterations')
plt.show()

