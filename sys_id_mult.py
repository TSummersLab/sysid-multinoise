import numpy as np
import numpy.random as npr
import numpy.linalg as la
import matplotlib.pyplot as plt

from matrixmath import specrad, mdot, vec, sympart, positive_semidefinite_part

########################################################################################################################
# Functions
########################################################################################################################

def groupdot(A,x):
    return np.einsum('...jk,...k',A,x)


def reshaper(X,m,n,p,q):
    Y = np.zeros([m*n,p*q])
    k = 0
    for j in range(n):
        for i in range(m):
            Y[k] = vec(X[i*p:(i+1)*p,j*q:(j+1)*q])
            k += 1
    return Y


def prettyprint(A,matname=None,fmt='%+13.9f'):
    print("%s = " % matname)
    if len(A.shape)==2:
        n = A.shape[0]
        m = A.shape[1]
        for icount,i in enumerate(A):
            print('[' if icount==0 else ' ', end='')
            print('[',end='')
            for jcount,j in enumerate(i):
                print(fmt % j,end=' ')
            print(']', end='')
            print(']' if icount==n-1 else '', end='')
            print('')


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


def example_system():
    n = 2
    m = 1
    A = np.array([[-0.2, 0.1],
                  [-0.4, 0.8]])
    B = np.array([[-1.8],
                  [-0.8]])
    SigmaA = 0.01*np.array([[ 0.8, -0.2,  0.0,  0.0],
                            [-0.2,  1.6,  0.2,  0.0],
                            [ 0.0,  0.2,  0.2,  0.0],
                            [ 0.0,  0.0,  0.0,  0.8]])
    SigmaB = 0.01*np.array([[0.5, -0.2],
                            [-0.2, 2.0]])
    return n,m,A,B,SigmaA,SigmaB


########################################################################################################################
# Main
########################################################################################################################

# for development only
plt.close()

seed = 0
npr.seed(seed)
# n,m,A,B,SigmaA,SigmaB = random_system(n=6,m=4,seed=seed)
n,m,A,B,SigmaA,SigmaB = example_system()

# Number of rollouts
nr = 100
# Rollout length
ell = 10000

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
x_hist[0] = npr.randn(nr,n)

# Generate the random input means and covariances
for t in range(ell):
    # Sample the means from a Gaussian distribution
    u_mean_hist[t] = 1*npr.randn(m)
    # Sample the covariances from a Wishart distribution
    u_covr_basevec = 0.01*npr.randn(m) # should the second dimension be 1 or > 1 ? does it matter?
    u_covr_hist[t] = np.outer(u_covr_basevec,u_covr_basevec)

# Generate the inputs
for t in range(ell):
    u_mean = u_mean_hist[t]
    u_covr = u_covr_hist[t]
    u_hist[t] = npr.multivariate_normal(u_mean, u_covr, nr)

# Generate the process noise
Anoise_vec_hist = npr.multivariate_normal(np.zeros(n*n), SigmaA,[ell,nr])
Bnoise_vec_hist = npr.multivariate_normal(np.zeros(n*m), SigmaB,[ell,nr])
Anoise_hist = np.reshape(Anoise_vec_hist,[ell,nr,n,n],order='F')
Bnoise_hist = np.reshape(Bnoise_vec_hist,[ell,nr,n,m],order='F')

# Collect rollout data
for t in range(ell):
    # Look up the previous state, input, and process noise
    x = x_hist[t]
    u = u_hist[t]
    Anoise = Anoise_hist[t]
    Bnoise = Bnoise_hist[t]
    # Transition the state
    x_hist[t+1] = groupdot(A+Anoise,x) + groupdot(B+Bnoise,u)

# First stage: mean dynamics parameter estimation
# Form data matrices for least-squares estimation
for t in range(ell+1):
    muhat_hist[t] = (1/nr)*np.sum(x_hist[t],axis=0)
    Xhat_hist[t] = (1/nr)*vec(np.sum(np.einsum('...i,...j',x_hist[t],x_hist[t]),axis=0))
    if t < ell:
        # What_hist[t] = (1/nr)*vec(np.sum(np.einsum('...i,...j',x_hist[t],u_mean_hist[t]),axis=0))
        What_hist[t] = vec(np.outer(muhat_hist[t],u_mean_hist[t]))
Y = muhat_hist[1:].T
Z = np.vstack([muhat_hist[0:-1].T,u_mean_hist.T])
# Solve least-squares problem
Thetahat = mdot(Y,Z.T,la.pinv(mdot(Z,Z.T)))
# Split learned model parameters
Ahat = Thetahat[:,0:n]
Bhat = Thetahat[:,n:n+m]

prettyprint(Ahat,"Ahat")
prettyprint(A,"A   ")
prettyprint(Bhat,"Bhat")
prettyprint(B,"B   ")

AAhat = np.kron(Ahat,Ahat)
ABhat = np.kron(Ahat,Bhat)
BAhat = np.kron(Bhat,Ahat)
BBhat = np.kron(Bhat,Bhat)

# Second stage: covariance dynamics parameter estimation
# Form data matrices for least-squares estimation
C = np.zeros([ell,n*n]).T
Uhat_hist = np.zeros([ell,m*m])
for t in range(ell):
    Uhat_hist[t] = vec(u_covr_hist[t] + np.outer(u_mean_hist[t], u_mean_hist[t]))
    Cminus = mdot(AAhat,Xhat_hist[t])+mdot(BAhat,What_hist[t])+mdot(ABhat,What_hist[t].T)+mdot(BBhat,Uhat_hist[t])
    C[:,t] = Xhat_hist[t+1] - Cminus
D = np.vstack([Xhat_hist[0:-1].T,Uhat_hist.T])
# Solve least-squares problem
SigmaThetahat_prime = mdot(C,D.T,la.pinv(mdot(D,D.T)))
# Split learned model parameters
SigmaAhat_prime = SigmaThetahat_prime[:,0:n*n]
SigmaBhat_prime = SigmaThetahat_prime[:,n*n:n*(n+m)]

# Reshape and project the noise covariance estimates onto the semidefinite cone
SigmaAhat = reshaper(SigmaAhat_prime,n,n,n,n)
SigmaBhat = reshaper(SigmaBhat_prime,n,m,n,m)
SigmaAhat = positive_semidefinite_part(SigmaAhat)
SigmaBhat = positive_semidefinite_part(SigmaBhat)

prettyprint(SigmaAhat,"SigmaAhat")
prettyprint(SigmaA,"SigmaA   ")
prettyprint(SigmaBhat,"SigmaBhat")
prettyprint(SigmaB,"SigmaB   ")


# Plotting
# Plot the rollout state data
if ell < 500:
    fig,ax = plt.subplots(n)
    plot_alpha = np.min([1,10/nr])
    for i in range(n):
        ax[i].step(t_hist,x_hist[:,:,i],color='tab:blue',linewidth=0.5,alpha=plot_alpha)
        ax[i].set_ylabel("State %d" % i)
    ax[n-1].set_xlabel("Time step")
    ax[0].set_title("Rollout data")
    plt.show()