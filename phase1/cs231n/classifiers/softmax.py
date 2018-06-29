import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_class = W.shape[1] 
  scores = X.dot(W)
# temp =np.zeros_like(W)*1
  for i in range(num_train):
    perImageScores = scores[i]
    # shifting the scores by substracting the max of each row
    perImageScores = perImageScores - np.max(perImageScores)
    loss += (-perImageScores[y[i]] + np.log(np.sum(np.exp(perImageScores))))
    for j in range(num_class):
        softmaxScore = np.exp(perImageScores[j]) / np.sum(np.exp(perImageScores))
        if j == y[i]:
#            temp[:,j] += -1 + softmaxScore
            dW[:,j] +=  (-1 + softmaxScore) * X[i]
        else:
#            temp[:,j] += softmaxScore
            dW[:,j] += softmaxScore * X[i]
 
  loss /= num_train
  loss += reg * np.sum(W*W)		
  dW /= num_train
  dW += 2* reg*W
 # print('dW from loop')
 # print(dW[range(0,500,100)])
  
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  temp = np.ones_like(X)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  # getting numercal stability
  shiftedScores = scores - np.max(scores,axis=1)[...,np.newaxis]
  # applying formula -f_i + log(summation of f_j)......done at for each row in shiftedscores
  loss = -shiftedScores[range(num_train),y] + np.log(np.sum(np.exp(shiftedScores),axis = 1))
  loss = loss.sum()
  loss /= num_train
  loss += reg*np.sum(W*W)
  # calulating gradient
  softmaxValues =np.exp(shiftedScores)/np.sum(np.exp(shiftedScores),axis=1)[...,np.newaxis]
 # print('softmax values before')
 # choice = np.random.choice(500,10)
 # print(softmaxValues[choice])	
  softmaxValues[range(num_train),y] -= 1
  #print('softmax values b')
 # print(softmaxValues[choice])	
 # print('temp from vector')
 # print(softmaxValues[range(0,500,100)])
  dW = np.dot(X.T,softmaxValues) 
  #temp2 = np.dot(temp.T,softmaxValues)
#  print('dW from vector')
#  print(dW[range(0,500,100)])
  dW /=num_train
  dW += 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

