#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad

def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """

    ### YOUR CODE HERE
    #print x
    x_squared = np.array(x) * np.array(x)
    sums = np.sum(x_squared, axis = 1)
    sums_length = np.sqrt(sums)

    shape = sums_length.shape
    new_shape = np.reshape(sums_length, (shape[0], 1))
    x = x / new_shape

    # shape = sums_length.shape
    # print shape
    # print np.reshape(sums_length, (shape[0], 1))

    ### END YOUR CODE

    return x


def test_normalize_rows():
    print "Testing normalizeRows..."
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]]))
    print x
    ans = np.array([[0.6,0.8],[0.4472136,0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print ""


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    predicted -- numpy ndarray, predicted word vector (\hat{v} in
                 the written component)
    target -- integer, the index of the target word
    outputVectors -- "output" vectors (as rows) for all tokens
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- cross entropy cost for the softmax word prediction
    gradPred -- the gradient with respect to the predicted word
           vector
    grad -- the gradient with respect to all the other word
           vectors

    We will not provide starter code for this function, but feel
    free to reference the code you previously wrote for this
    assignment!
    """

    ### YOUR CODE HERE
    #length = predicted.len
    # one_hot = np.zeros(length)
    # one_hot[target] = 1
    #
    # y_hat = outputVectors * predicted
    # denom = np.sum(y_hat)
    #
    # y_hat = y_hat / denom

    y_hat = softmax(np.dot(outputVectors, predicted))

    #y_hat is fine


    cost = -1 * np.log(y_hat[target]) #cuz you only care about the target, probably need a summation somewhere

    delta = y_hat
    delta[target] = delta[target] - 1

    delta_shape = delta.shape
    delta = np.reshape(delta, (delta_shape[0], 1))


    #print delta, outputVectors, predicted

    #print np.reshape(delta, (5,1))

    gradPred = (outputVectors.T.dot(delta).T)[0]
    grad = predicted * delta # rn its doing an outer product WHICH IS THE RIGHT THING WOO HOO

    #print grad
    #print grad
    ### END YOUR CODE

    return cost, gradPred, grad


def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target """

    indices = [None] * K
    for k in xrange(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))

    #Indicies now tell us which indices we treat as our outputVectors

    ### YOUR CODE HERE

    words = outputVectors[indices]

    # print newVectors
    #new Vectors has length 11. Looks right

    labels = np.array([1] + [-1 for k in range(K)])
    #print signs
    newVectors = (np.dot(words, predicted)) * labels
    # print newVectors
    # print 'done'
    # new Vectors is thus the correct things for the insides of our vector

    z = np.dot(words, predicted) * labels
    probs = sigmoid(z)
    cost = - np.sum(np.log(probs))

    dx = (probs - 1) * labels

    gradPred = dx.reshape((1, K + 1)).dot(words).flatten()
    #print gradPred.shape

    grad_temp = dx.reshape((K + 1, 1)).dot(predicted.reshape(1, predicted.shape[0])) # this thing is perfect

    grad = np.zeros(outputVectors.shape)
    grad1 = np.zeros(outputVectors.shape)

    #print indices, grad, grad_temp

    #FUCK LA SO IT ALL CAME DOWN TO THIS FOR LOOP. WHATS THE DIFFERENCE BETWEEN THIS AND THE WAY I HAD IT????
    grad1[indices] += grad_temp # I guess cannot be vectorized because of the shared indices i guess
    #print grad1

    for k in range(K + 1):
        grad[indices[k]] += grad_temp[k, :]

    #print grad
    #print "end"
    #
    # #print grad somethign is fucky but idk whats
    # #print grad.shape, outputVectors.shape
    # # GRAD IS UNFINISHED OK
    # ### END YOUR CODE

    return cost, gradPred, grad

def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currrentWord -- a string of the current center word
    C -- integer, context size
    contextWords -- list of no more than 2*C strings, the context words
    tokens -- a dictionary that maps words to their indices in
              the word vector list
    inputVectors -- "input" word vectors (as rows) for all tokens
    outputVectors -- "output" word vectors (as rows) for all tokens
    word2vecCostAndGradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    #print tokens, currentWord, contextWords
    #print inputVectors, outputVectors # so input and output vectors have the same dimensions. But when to use one or the other
    curr_token = tokens[currentWord]
    curr_vec = inputVectors[curr_token]
    #print curr_vec

    context_tokens = [tokens[i] for i in contextWords]
    #context_words = [outputVectors[i] for i in context_tokens]
    #print context_vec

    for token in context_tokens:
        cost_temp, gradPred, grad = word2vecCostAndGradient(curr_vec, token, outputVectors, dataset)
        cost += cost_temp
        gradOut += grad
        gradIn[curr_token] += gradPred


    # SO TO GET THE RIGHT GRADIENTS YOU NEED TO DO SOMETHING DIFFERENT Hmm Cuz these are each matrices for each of the positions

    #cost, gradPred, grad =  word2vecCostAndGradient(curr_vec, context_tokens, outputVectors, dataset)

    return cost, gradIn, gradOut


def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """CBOW model in word2vec

    Implement the continuous bag-of-words model in this function.

    Arguments/Return specifications: same as the skip-gram model

    Extra credit: Implementing CBOW is optional, but the gradient
    derivations are not. If you decide not to implement CBOW, remove
    the NotImplementedError.
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    context_tokens = [tokens[i] for i in contextWords]
    context_words = [inputVectors[i] for i in context_tokens]

    v_hat = np.sum(context_words, axis = 0)

    target_context = tokens[currentWord]



    cost, gradPred, gradOut = word2vecCostAndGradient(v_hat, target_context, outputVectors, dataset)

    # print 'I"MG HERE'
    # print grad, gradPred
    # print 'STOP'

    #print grad.shape, gradPred.shape, gradIn.shape, gradOut.shape

    #gradOut = gradPred

    for token in context_tokens :
        gradIn[token] += gradPred # for some reasonits additive. DOn't ask me why ???? I guess when we update these parameters

    #print gradIn

    return cost, gradIn, gradOut


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    for i in xrange(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom

        # so in negsamping its our input vectors that have some issue. the value is waayyy too small

    return cost, grad


def test_word2vec():
    """ Interface to the dataset for negative sampling """
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in xrange(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])

    #print getNegativeSamples(1, dataset, 10) THIS ONE GIVES US 10 random indices to us

    print "==== Gradient check for skip-gram ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)
    print "\n==== Gradient check for CBOW      ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)

    print "\n=== Results ==="
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print skipgram("c", 1, ["a", "b"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient)
    print cbow("a", 2, ["a", "b", "c", "a"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print cbow("a", 2, ["a", "b", "a", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient)


if __name__ == "__main__":
    test_normalize_rows()

    test_word2vec()