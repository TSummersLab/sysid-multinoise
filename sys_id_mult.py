import numpy as np
import numpy.random as npr
import numpy.linalg as la
import matplotlib.pyplot as plt

from matrixmath import specrad, mdot, vec, sympart

npr.seed(2)

# Problem data
n = 2
m = 1

A = npr.randn(n,n)
A = A*0.5/specrad(A)
B = npr.randn(n,m)

SigmaA_basevec = 0.1*npr.randn(n*n)
SigmaB_basevec = 0.1*npr.randn(n*m)
SigmaA = np.outer(SigmaA_basevec,SigmaA_basevec)
SigmaB = np.outer(SigmaB_basevec,SigmaB_basevec)

# Number of rollouts
nr = 100

# Rollout length
ell = 1000

# Preallocate history matrices
t_hist = np.arange(ell+1)
x_hist = np.zeros([ell+1,nr,n])
u_hist = np.zeros([ell,nr,m])
u_mean_hist = np.zeros([ell,m])
u_covr_hist = np.zeros([ell,m,m])
muhat_hist = np.zeros([ell+1,n])
Xhat_hist = np.zeros([ell+1,n*n])
What_hist = np.zeros([ell+1,n*m])
Chat_hist = np.zeros([ell+1,n*n])

# Initialize the state
x0 = npr.randn(n)
for k in range(nr):
    x_hist[0,k] = x0

# Generate the random input means and covariances
# todo - vectorize
for t in range(ell):
    u_mean_hist[t] = 0.1*npr.randn(m)
    u_covr_basevec = 0.1*npr.randn(m) # should the second dimension be 1 or > 1 ? does it matter?
    u_covr_hist[t] = np.outer(u_covr_basevec,u_covr_basevec)

# Collect rollout data
# todo - vectorize so all rollouts / state transitions happen at once via matrix multiplication
for k in range(nr):
    x = x0
    for t in range(ell):
        # Generate the random input
        # todo - vectorize and pregenerate
        u_mean = u_mean_hist[t]
        u_covr = u_covr_hist[t]
        u = npr.multivariate_normal(u_mean, u_covr)

        # Generate the random noise
        Anoise_vec = npr.multivariate_normal(np.zeros(n*n), SigmaA)
        Bnoise_vec = npr.multivariate_normal(np.zeros(n*m), SigmaB)

        Anoise = np.reshape(Anoise_vec,[n,n])
        Bnoise = np.reshape(Bnoise_vec,[n,m])

        # Transition the state
        x = mdot((A+Anoise),x) + mdot((B+Bnoise),u)

        # Record quantities
        x_hist[t+1,k] = x
        u_hist[t,k] = u

plt.step(t_hist,x_hist[:,:,0],color='tab:blue',linewidth=0.5,alpha=0.5)

# First stage: mean dynamics parameter estimation

# Form data matrices for least-squares estimation
for t in range(ell+1):
    muhat_hist[t] = (1/nr)*np.sum(x_hist[t],axis=0)
    Xhat_hist[t] = (1/nr)*vec(np.sum(np.einsum('...i,...j',x_hist[t],x_hist[t]),axis=0))
    if t < ell:
        What_hist[t] = (1/nr)*vec(np.sum(np.einsum('...i,...j',x_hist[t],u_mean_hist[t]),axis=0))
Y = muhat_hist[1:].T
Z = np.vstack([muhat_hist[0:-1].T,u_mean_hist.T])

# Solve least-squares problem
Thetahat = mdot(Y,Z.T,la.pinv(mdot(Z,Z.T)))
Ahat = Thetahat[:,0:n]
Bhat = Thetahat[:,n:n+m]

print(Ahat)
print(A)
print(Bhat)
print(B)

AAhat = np.kron(Ahat,Ahat)
ABhat = np.kron(Ahat,Bhat)
BAhat = np.kron(Bhat,Ahat)
BBhat = np.kron(Bhat,Bhat)

# Second stage: covariance dynamics parameter estimation
C = np.zeros([ell,n*n]).T
Uhat_hist = np.zeros([ell,m*m])
for t in range(ell):
    Uhat_hist[t] = vec(u_covr_hist[t])
    Cminus = mdot(AAhat,Xhat_hist[t])+mdot(BAhat,What_hist[t])+mdot(ABhat,What_hist[t].T)+mdot(BBhat,Uhat_hist[t])
    C[:,t] = Xhat_hist[t+1] - Cminus
D = np.vstack([Xhat_hist[0:-1].T,Uhat_hist.T])
SigmaThetahat_blk = mdot(C,D.T,la.pinv(mdot(D,D.T)))
SigmaAhat_blk = SigmaThetahat_blk[:,0:n*n]
SigmaBhat_blk = SigmaThetahat_blk[:,n*n:n*(n+m)]

def reshaper(X,m,n,p,q):
    Y = np.zeros([m*n,p*q])
    k = 0
    for j in range(n):
        for i in range(m):
            Y[k] = vec(X[i*p:(i+1)*p,j*q:(j+1)*q])
            k += 1
    return Y

def positive_semidefinite_part(X):
    X = sympart(X)
    n = X.shape[0]
    Y = np.zeros_like(X)
    eigvals, eigvecs = la.eig(X)
    for i in range(n):
        if eigvals[i] > 0:
            Y += eigvals[i]*np.outer(eigvecs[i],eigvecs[i])
    return Y

# Reshape and project the noise covariance estimates onto the semidefinite cone
SigmaAhat = positive_semidefinite_part(reshaper(SigmaAhat_blk,n,n,n,n))
SigmaBhat = positive_semidefinite_part(reshaper(SigmaBhat_blk,n,m,n,m))

print(SigmaAhat)
print(SigmaA)
print(SigmaBhat)
print(SigmaB)