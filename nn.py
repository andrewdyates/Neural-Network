#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright © 2011 Andrew D. Yates
# andrewyates.name@gmail.com

import math
import random
# ideally use numpy for faster matrix math
# import numpy

PN_4BIT_PARITY = [
  [[0,0,0,0], 0],
  [[0,0,0,1], 1],
  [[0,0,1,0], 1],
  [[0,0,1,1], 0],
  [[0,1,0,0], 1],
  [[0,1,0,1], 0],
  [[0,1,1,0], 0],
  [[0,1,1,1], 1],
  [[1,0,0,0], 1],
  [[1,0,0,1], 0],
  [[1,0,1,0], 0],
  [[1,0,1,1], 1],
  [[1,1,0,0], 0],
  [[1,1,0,1], 1],
  [[1,1,1,0], 1],
  [[1,1,1,1], 0],
]

def rand():
  """Return a random number between -1 and 1."""
  random.seed()
  r = random.uniform(-1, 1)
  return r


class StaticNetwork(object):
  def __init__(self, eta=0.05, alpha=0):
    """Initialize a fully connected 4-1-1 neural network.

    Args:
      eta: training rate for all neurons
      alpha: momentum for all neurons

    This class should be dynamically populated to handle other
    networks in the future.
    """
    self.hidden = [
      Neuron(degree=4, eta=eta, alpha=alpha),
      Neuron(degree=4, eta=eta, alpha=alpha),
      Neuron(degree=4, eta=eta, alpha=alpha),
      Neuron(degree=4, eta=eta, alpha=alpha),
      ]
    self.output = Neuron(degree=4, eta=eta, alpha=alpha)

  def forward(self, input):
    """Complete one forward pass through the network.

    Args:
      input: [num]; len(input) = 4
    Returns:
      num y of neural network output
    """
    next_input = []
    for n in self.hidden:
      x = n.forward(input)
      next_input.append(x)
    y = self.output.forward(next_input)
    return y

  def backward(self, y, d):
    """Complete one backwards pass through the network.

    Args:
      y: num of actual output from last forward pass
      d: num of expected output for last forward pass
    """
    e = d - y
    self.output.backward(e)
    for i in range(4):
      e = self.output.weighted_grad[i]
      self.hidden[i].backward(e)

  def run(self, input, d):
    """Complete one network pass.

    Args:
      input: [num] of input, len(input) = 4
      d: num desired output
    Returns:
      num error d - y
    """
    y = self.forward(input)
    self.backward(y, d)
    e = d - y
    return e

  def train(self, epoch):
    """Complete one training epoch.

    Args:
      epoch: [[num], num] of input, output pairs
    """
    es = []
    for input, d in epoch:
      e = self.run(input, d)
      es.append(abs(e))
    avg_e = sum(es) / len(epoch)
    max_e = max(es)
    return avg_e, max_e

  def test(self, epoch):
    """Run network forward over epoch and compare outputs.
 
    Args:
      epoch: [[num], num] of input, output pairs
    Returns:
      [[num], num, num] of input, expected, actual triplets
    """
    output = []
    for input, d in epoch:
      y = self.forward(input)
      output.append([input, d, y])
    return output
  
      
class Neuron(object):
  def __init__(self, degree, eta=0.05, a=1, alpha=0):
    """Initialize neural network neuron.

    Args:
      degree: int >0 of inputs to neuron
      eta: num >0 learning rate
      a: num >0 of activation threshold
      alpha: num >=0 of weight update momentum
    """
    self.degree = degree
    self.eta = eta
    self.a = a
    self.alpha = alpha
    # initalize weights plus bias weight as last weight
    self.weight = [rand()]*(degree+1)
    # initialize null vectors as None
    self.input = None
    self.output = None
    self.weighted_grad = None
    self.last_delta_w = None

  def forward(self, input):
    """Compute function signal for this neuron.

    Args:
      input: [num] of input vector to neuron; len(x) == self.degree
    Returns:
      num of self.output
    """
    # copy input vector `x` and add constant weight scalar "1" to end
    self.input = list(input)
    self.input.append(1)

    # sum weighted input `v(n)`
    v = sum([x*w for x, w in zip(self.input, self.weight)])
    # compute activation `y(n)` from weighted sum of inputs 
    self.output = self.phi(v)
    return self.output


  def backward(self, error):
    """Compute gradient and weight changes for this neuron.

    Note: the network must provide the error term based on other neurons.
    For output node, error is the desired output minus the actual output
      delta(n) = e(n) = d(n) - y(n)
    For hidden node "j", error is the weighted sum of the next layer "k" errors
      delta(n) = SUM(error_k * weight_j_k)

    Args:
      error: num of error; either e(n) or weighted sum of errors from next layer
    """
    local_grad = error * self.d_phi(self.output)
    # save weighted gradient for next backward pass layer
    self.weighted_grad = map(lambda x: local_grad * x, self.weight)
    
    # compute weight delta 
    delta_w = [self.eta * local_grad * x for x in self.input]
    # compute momentum; if no momentum, then momentum is zero
    if self.last_delta_w:
      momentum_w = [self.alpha * w for w in self.last_delta_w]
    else:
      momentum_w = [0]*len(self.weight)
    # update weight
    new_weight = []
    for w, d_w, m_w in zip(self.weight, delta_w, momentum_w):
      new_weight.append(w + d_w + m_w)
    self.weight = new_weight
    # save delta_w for next iteration
    self.last_delta_w = delta_w

  def phi(self, v):
    """Activation function.

    Args:
      v: num of vector sum weights * inputs, sum(w*x)
    Returns:
      num `y` of activation output
    """
    y = 1 / (1 + math.exp(-self.a * v))
    return y

  def d_phi(self, y):
    """Derivative of activation function `phi`.

    Args:
      y: num of the activation output, y = phi(v)
    Returns:
      num of the derivative of the activation funct phi
    """
    dy = self.a * y * (1 - y)
    return dy
