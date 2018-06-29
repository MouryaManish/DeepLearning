import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0

  for i in range(num_train):
    count = 0
    scores = X[i].dot(W) #[10 scores in a row for each class]
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
          count += 1
          loss += margin
          dW[:,j] += X[i]

    dW[:,y[i]] += -1*count*X[i] 
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  # regularization of gradient
  dW = dW /num_train + 2*reg *W 
	
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  count =0.0
  loss = 0.0
  num_train = X.shape[0]
  dW = np.zeros(W.shape) # initialize the gradient as zero
  scores = X.dot(W)
  correct_score = scores[[i for i in range(X.shape[0])],y]
 # print('correct score ',correct_score)
  mask = np.ones(scores.shape,dtype=bool)
  mask[[i for i in range(scores.shape[0])],y]=False
  #print('mask ',mask[range(1,500,50)])
  newScores = scores[mask].reshape(scores.shape[0],scores.shape[1]-1)
  correct_score = np.expand_dims(correct_score, axis=1)
  lossMatrix = newScores - correct_score + 1 
  #print('lossMatrix ',lossMatrix[range(1,500,50)])
  temp=np.greater(lossMatrix,0)
 # print('temp shape', temp.shape)
 # print('temp ',temp[range(1,500,50)])
  
  lossMatrix = lossMatrix[temp]
  loss = np.sum(lossMatrix)/num_train
  loss += reg * np.sum(W*W)
 # print(temp[...,1]) 

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  lossMatrix = scores - correct_score + 1 
  temp=np.greater(lossMatrix,0)
  count = temp.sum(1) - 1
  temp = temp.astype(float)

  temp[[i for i in range(temp.shape[0])],y] = -1*count
  dW = np.dot(X.T,temp)
  dW = dW /num_train + 2*reg *W 
  
  return loss, dW
