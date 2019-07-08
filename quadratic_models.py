import numpy as np
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import SGD, Adam
from keras.layers import Activation
from keras import backend as K
from keras.layers import Layer
from keras.utils.generic_utils import get_custom_objects

def squared_activation(x):
    return K.square(x)
get_custom_objects().update({'square_activation': Activation(squared_activation)})

class Quadratic(Layer):
    """
    A Keras Quadratic layer.
    """

    def __init__(self, output_dim, kernel_initializer='glorot_normal', **kwargs):
        self.output_dim = output_dim
        self.kernel_initializer = kernel_initializer
        super(Quadratic, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[-1], input_shape[-1], self.output_dim),
                                      initializer=self.kernel_initializer,
                                      trainable=True)
        super(Quadratic, self).build(input_shape)

    def call(self, x, **kwargs):
        first = K.dot(x, self.kernel)
        return K.batch_dot(K.permute_dimensions(first,(0,2,1)), x)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


class Square(Layer):
    """
    A Keras squared layer (activation function).
    """

    def __init__(self, **kwargs):
        super(Square, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Square, self).build(input_shape)

    def call(self, x, **kwargs):
        return K.square(x)

    def compute_output_shape(self, input_shape):
        return input_shape


def sig(x):
    """
    The sigmoid function.
    """
    x[x>10] = 10
    x[x<-10] = -10
    return 1. / (1. + np.exp(-x))


def softmax(x):
    """
    The softmax function.
    """
    x -= np.max(x, axis=1, keepdims=True)
    x[x<-10] = -10
    x = np.exp(x)
    x /= np.sum(x, axis=1, keepdims=True)
    return x


def project_GLRAM(As, r, iter=25, verbose=False):
    """
    Given a set of matrices and a desired lower dimension, project the matrices
    using GLRAM.
    :param As: the input matrices
    :param r: the lower dimension
    :param iter: number of GLRAM iterations
    :param verbose: verbosity
    :return: The U matrix and the set of lower dimensional S matrices
    """

    d = As.shape[-1]
    label_num = As.shape[0]

    # get the first matrix to decompose:
    M = np.zeros((d,d))
    for c in range(label_num):
        M += As[c,:,:] @ As[c,:,:]
    temp = M[:]

    # get the top-r eigenvectors:
    _, vecs = np.linalg.eigh(M)
    U = vecs[:, -r:]

    # iterate:
    for i in range(iter-1):
        M = np.zeros((d, d))
        for c in range(label_num):
            M += As[c, :, :] @ U @ U.T @ As[c, :, :]
        if verbose:
            print("Difference: " + str(np.linalg.norm(temp-M)))
        temp = M[:]

        _, vecs = np.linalg.eigh(M)
        U = vecs[:, -r:]

    # recover the S matrices:
    S = np.zeros((r, r, label_num))
    for c in range(label_num):
        S[:,:,c] = U.T @ As[c,:,:] @ U

    return U, S


class binary_squared_model:
    """
    The binary squared model from the blog, with low rank projections.
    """

    def __init__(self, d, r, lr):

        self.W = np.zeros((d,r))
        self.alpha = np.zeros(r)
        self.lr = lr
        self.d = d
        self.r = r

    def predict(self, x):
        h = np.square(x @ self.W)
        logits = h @ self.alpha
        return sig(logits)

    def train_on_batch(self, x, y):

        # calculate error signals
        pred = self.predict(x)
        err = pred.squeeze() - y

        # calculate the change to the quadratic variables:
        b = x.shape[0]
        mat_w = np.zeros((self.d, self.d))
        for i in range(self.r):
            mat_w += self.alpha[i] * self.W[:,i].reshape((self.d,1)) @ self.W[:,i].reshape((1,self.d))
        mat_x = np.zeros(mat_w.shape)
        for j in range(b):
            mat_x -= err[j] * x[j,:].reshape((self.d,1)) @ x[j,:].reshape((1,self.d))
        mat_x *= self.lr / b
        mat = mat_x + mat_w
        vals, vecs = np.linalg.eigh(mat)
        abs_vals = np.abs(vals)
        indices = np.flip(np.argsort(abs_vals), axis=0)[:self.r]

        # update quadratic variables:
        self.W = vecs[:,indices]
        self.alpha = vals[indices]

    def evaluate(self, x, y):
        pred = self.predict(x)
        pred[pred<=0.5] = 0
        pred[pred>0.5] = 1
        return np.mean(pred == y)

    def get_norm(self):
        mat = np.zeros((self.W.shape[0],self.W.shape[0]))
        for i in range(self.W.shape[1]):
            mat += self.alpha[i]*self.W[:,i].reshape(-1,1) @ self.W[:,i].reshape(1,-1)
        return np.linalg.norm(mat)


def binary_SGD_squared_model(d, r, lr):
    """
    the binary squared model from the blog, optimized with regular SGD.
    :param d: input dimension
    :param r: lower dimension
    :param lr: learning rate
    :return: a compiled Keras binary squared model
    """

    i = Input(shape=(d,))
    quad = Dense(r, activation=squared_activation, use_bias=False, kernel_initializer='glorot_normal', name='W')(i)
    out = Dense(1, use_bias=False, name='alpha')(quad)
    out = Activation('sigmoid')(out)
    model = Model(inputs=[i], outputs=[out])
    model.compile(optimizer=SGD(lr), loss='binary_crossentropy',  metrics=['acc'])

    return model


class GLRAM_quadratic_model:
    """
    The new quadratic model from the blog, optimized with GLRAM.
    """

    def __init__(self, d, r, lr, label_num=10):

        self.U = np.zeros((d,r))
        self.S = np.zeros((r, r, label_num))
        self.lr = lr
        self.d = d
        self.r = r
        self.label_num = label_num

    def predict(self, x):

        # quadratic layer:
        reduced = x @ self.U
        quad = np.tensordot(reduced, self.S, ((1),(0)))
        logits = np.einsum("br,brc->bc", reduced, quad)

        return softmax(logits)

    def train_on_batch(self, x, y):

        # calculate error signals
        pred = self.predict(x)
        err = pred - y

        # calculate the current quadratic variables in canonical form:
        mats_A = np.zeros((self.label_num, self.d, self.d))
        for c in range(self.label_num):
            mats_A[c, :, :] = self.U @ self.S[:,:,c] @ self.U.T

        # calculate the gradients for every matrix:
        xx = np.einsum("ij,ik->ijk", x, x)
        grads_A = np.einsum("im,ijk->mjk", err, xx) / xx.shape[0]
        mats_A -= self.lr * grads_A

        # project back onto network variables:
        self.U, self.S = project_GLRAM(mats_A, self.r)

    def evaluate(self, x, y):
        pred = self.predict(x)
        return np.mean(np.argmax(pred, axis=1) == np.argmax(y, axis=1))

    def get_norm(self):

        norms = np.zeros(self.label_num)
        for c in range(self.label_num):
            norms[c] = np.linalg.norm(self.U @ self.S[:, :, c] @ self.U.T)
        return np.mean(norms)


def SGD_quadratic_model(d, r, lr, label_num = 10):
    """
    the quadratic model from the blog, optimized with regular SGD.
    :param d: input dimension
    :param r: lower dimension
    :param lr: learning rate
    :param label_num: the number of labels
    :return: a compiled Keras quadratic model
    """

    i = Input(shape=(d,))
    dense = Dense(r, use_bias=False, kernel_initializer='glorot_normal', name='U')(i)
    quad = Quadratic(label_num, name='S')(dense)
    out = quad
    out = Activation('softmax')(out)
    model = Model(inputs=[i], outputs=[out])
    model.compile(optimizer=SGD(lr), loss='categorical_crossentropy',  metrics=['acc'])

    return model


def SGD_squared_model(d, r, lr, label_num = 10):
    """
    the "old" quadratic model from the blog, optimized with regular SGD.
    :param d: input dimension
    :param r: lower dimension
    :param lr: learning rate
    :param label_num: the number of labels
    :return: a compiled Keras squared model
    """

    i = Input(shape=(d,))
    quad = Dense(r, activation=squared_activation, use_bias=False, name='U')(i)
    out = Dense(label_num, use_bias=False, name='alpha')(quad)
    out = Activation('softmax')(out)
    model = Model(inputs=[i], outputs=[out])
    model.compile(optimizer=SGD(lr), loss='categorical_crossentropy',  metrics=['acc'])

    return model


def get_glram_frob_norm(model):
    U = model.get_layer('U').get_weights()[0]
    S = model.get_layer('S').get_weights()[0]
    norms = np.zeros(S.shape[-1])
    for c in range(S.shape[-1]):
        norms[c] = np.linalg.norm(U @ S[:,:,c] @ U.T)
    return np.mean(norms)

def get_binary_frob_norm(model):
    W = model.get_layer('W').get_weights()[0]
    alpha = model.get_layer('alpha').get_weights()[0]
    mat = np.zeros((W.shape[0], W.shape[0]))
    for i in range(W.shape[1]):
        mat += alpha[i] * W[:, i].reshape(-1, 1) @ W[:, i].reshape(1, -1)
    return np.linalg.norm(mat)

