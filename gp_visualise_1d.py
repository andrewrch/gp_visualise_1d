#!/usr/bin/env python
#
# This code is heavily based on code used during video lectures from
# Mathematical Monk's videos about Gaussian Processes on youtube
#
# https://www.youtube.com/user/mathematicalmonk
#
# I am just reimplementing it in an attempt to understand it better

import numpy as np
import matplotlib.pyplot as plot
import argparse


def cov_function(c):
  """ Returns a covariance function given an int input"""
  return {
      '1': lambda x, y : 1 * (x.T * y),
      '2': lambda x, y : 1 * min(x, y),
      '3': lambda x, y : np.exp(-100 * ((x - y).T * (x - y)))
  }[str(c)]

def main():
  parser = argparse.ArgumentParser(description='Script to draw functions from '\
                                               'a Gaussian Process')
  parser.add_argument('-x', type=int, nargs=2, default=(0,5))
  parser.add_argument('-s', type=int, default=100)
  parser.add_argument('-c', type=int, default=1)
  args = parser.parse_args()

  # Sample values
  x = np.linspace(args.x[0], args.x[1], num=args.s)
  n = len(x)
  # Covariance matrix
  c = np.zeros((n, n))
  # Select covariance function
  k = cov_function(args.c)

  # Build a covariance matrix using x values and the covariance function
  for i in range(n):
    for j in range(n):
      c[i, j] = k(x[i], x[j])

  # Now we randomly sample from the distribution made by the covariance matrix
  u = np.random.randn(n, 1)
  A, s, B = np.linalg.svd(c)
  S = np.diag(s)
  z = A.dot(np.sqrt(S).dot(u))

  plot.plot(x, z)
  plot.show()

if __name__=='__main__':
  main()
