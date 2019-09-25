import numpy as np
import numpy.random as npr

from matrixmath import specrad

def random_system(n=2,m=1,seed=0):
    npr.seed(seed)
    A = npr.randn(n,n)
    A = A*0.8/specrad(A)
    B = npr.randn(n,m)
    SigmaA_basevec = 0.1*npr.randn(n*n)
    SigmaB_basevec = 0.1*npr.randn(n*m)
    SigmaA = np.outer(SigmaA_basevec,SigmaA_basevec)
    SigmaB = np.outer(SigmaB_basevec,SigmaB_basevec)
    return n,m,A,B,SigmaA,SigmaB


def example_system_scalar(Sa=0.5,Sb=0.5):
    n = 1
    m = 1
    A = np.array([[-0.5]])
    B = np.array([[1]])
    SigmaA = np.array([[Sa]])
    SigmaB = np.array([[Sb]])
    return n,m,A,B,SigmaA,SigmaB


def example_system_twostate():
    n = 2
    m = 1
    A = np.array([[-0.2, 0.3],
                  [-0.4, 0.8]])
    B = np.array([[-1.8],
                  [-0.8]])
    SigmaA = 0.1*np.array([[  0.8, -0.2,  0.0,  0.0],
                            [-0.2,  1.6,  0.2,  0.0],
                            [ 0.0,  0.2,  0.2,  0.0],
                            [ 0.0,  0.0,  0.0,  0.8]])
    SigmaB = 0.1*np.array([[ 0.5, -0.2],
                           [-0.2,  2.0]])
    return n,m,A,B,SigmaA,SigmaB

def example_system_twostate_diagonal():
    n = 2
    m = 2
    A = np.array([[-0.2, 0.3],
                  [-0.4, 0.8]])
    B = np.array([[-1.8, 0.3],
                  [-0.8, 0.6]])
    SigmaA = 0.1*np.eye(n*n)
    SigmaB = 0.1*np.eye(n*m)
    return n,m,A,B,SigmaA,SigmaB