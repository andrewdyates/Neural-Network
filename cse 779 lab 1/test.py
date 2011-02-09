#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright © 2011 Andrew D. Yates
# andrewyates.name@gmail.com
"""Test neural network for 4 bit parity function. Print output to stdout."""

import nn

MAX_I = 4000000

def main():
  """Test network with variable parameters."""
  # results is an array of tuples
  results = []
  # run simulation
  for i in range(0,2):
    for j in range(1, 11):
      alpha = i * 0.9
      eta = j * 0.05
      # run each test three times
      for k in range(5):
        epochs = test_net(eta, alpha)
        # save results as tuple
        results.append([eta, alpha, epochs])
        
  # print results
  print "######"
  print "Result Table for Network Lab"
  print "eta, alpha, epochs"
  print "=================="
  for eta, alpha, epochs in results:
    print "%1.2f, %1.2f, %d" % (eta, alpha, epochs)


def test_net(eta, alpha):
  """Test new Neural Network for parameters.

  Prints to STDOUT.

  Args:
     eta: num >0 of learning rate
     alpha num >= of momentum scalar
  Returns:
     int of epochs until trained or None if failed to converge
  """
  net = nn.StaticNetwork(eta=eta, alpha=alpha)
  # complete one epoch training
  print "Begin Training for eta=%1.2f, alpha=%1.2f" % (eta, alpha)
  print "==="
  for i in range(MAX_I):
    avg_e, max_e = net.train(nn.PN_4BIT_PARITY)
    if max_e <= 0.05:
      print "======"
      print "TRAINING COMPLETED AFTER %d EPOCHS." % i
      print "NETWORK: eta=%1.2f, alpha=%1.2f" % (eta, alpha)
      print "#: %d, E: %6f" % (i, avg_e)
      print "Max error: %6f" % max_e
      print "Average error: %6f" % avg_e
      print_status(net)
      return i
    # print progress
    if i % 4000 == 0:
      print "#i: %d, avg_E: %1.6f, max_E: %1.6f" % (i, avg_e, max_e)
    if i % 30000 == 0:
      print_status(net)
      
  # network failed to converge!
  print "Network failed to converge after maximum %d iterations!" % MAX_I
  print_status(net)
  return None
  
def print_status(net):
  for input, d, y in net.test(nn.PN_4BIT_PARITY):
    print "%s => %d ~= %1.6f, D: %1.3f" % (input, d, y, abs(d-y))
  
if __name__ == '__main__':
  main()
