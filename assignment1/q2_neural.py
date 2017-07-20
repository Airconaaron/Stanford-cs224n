#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    #print W2.shape, b2.shape

    #print len(W1), data[0], b1, len(data)

    ### YOUR CODE HERE: forward propagation
    #Eg, find the cost function. Save some intermediate stuff though, seems like it'd be useful
    #h = sigmoid(x * w1 + b1)
    # y = (softmax( h * w2 + b2)
    # hence  the cost function will be labels * log(y) and then sum it all up

    z_1 = np.matrix(data) * W1 + b1
    h = sigmoid(z_1)
    y_prime = softmax(h * W2 + b2)
    logs = np.log(y_prime)

    #print y_prime.shape

    #print np.array(logs) * labels

    cost = - np.sum(np.array(logs) * labels, axis = 1)
    cost = np.sum(cost) # lets add up each instance fo the cost for now and see what happens

    # My question is then do we just sum up the costs of each function
    #print cost #somethign is printing so I'm gonan say i'm a genius right here duh

    #Cost(y, y') = -sum of (y * log Y')
    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation

    # you'll need gradients for each parameter except for the input vectors. Right now this isn't even a word2vec
    delta_1 = y_prime - labels
    delta_2 = delta_1 * W2.T
    #print sigmoid_grad(h).shape
    delta_3 = np.array(delta_2) * sigmoid_grad(h)

    gradW2 = np.array(h.T * delta_1) # i dunno or its reverse OMG I HASTE EVERYONE why is it that np.array fixes everything. Sigh
    gradb2 = np.array(np.sum(delta_1, axis=0)) # main issue is that this is a 20 x 5 vector when it should be a 1 x 5
    gradW1 = data.T.dot(delta_3)
    gradb1 = np.sum(delta_3, axis=0) # this should be 1 x10 not 20  x 5



    ### END YOUR CODE

    #print gradW1, gradW1.flatten()
    # print 'jee'

    ### Stack gradients (do not modify)
    grad = np.concatenate((
        gradW1.flatten(),
        gradb1.flatten(),
        gradW2.flatten(),
        gradb2.flatten())
        )
    #print grad
    #print cost
    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    forward_backward_prop(data, labels, params, dimensions)

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)




def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
