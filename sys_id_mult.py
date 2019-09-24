import numpy as np
import numpy.random as npr
import numpy.linalg as la

from matrixmath import specrad, mdot, vec, sympart, positive_semidefinite_part

from time import time

from system_definitions import random_system, example_system_scalar, example_system_twostate

from plotting import plot_trajectories,plot_model_estimates,plot_estimation_error,plot_estimation_error_multi


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


def generate_sample_data(n,m,SigmaA,SigmaB,nr,ell,u_mean_var=1.0,u_covr_var=0.1):
    # Generate the random input means and covariances
    u_mean_hist = np.zeros([ell, m])
    u_covr_hist = np.zeros([ell, m, m])
    for t in range(ell):
        # Sample the means from a Gaussian distribution
        u_mean_hist[t] = u_mean_var*npr.randn(m)
        # Sample the covariances from a Wishart distribution
        u_covr_base = u_covr_var*npr.randn(m,m)
        u_covr_hist[t] = np.dot(u_covr_base.T,u_covr_base)

    # Generate the inputs
    u_hist = np.zeros([ell, nr, m])
    for t in range(ell):
        u_mean = u_mean_hist[t]
        u_covr = u_covr_hist[t]
        u_hist[t] = npr.multivariate_normal(u_mean, u_covr, nr)

    # Generate the process noise
    Anoise_vec_hist = npr.multivariate_normal(np.zeros(n * n), SigmaA, [ell, nr])
    Bnoise_vec_hist = npr.multivariate_normal(np.zeros(n * m), SigmaB, [ell, nr])
    Anoise_hist = np.reshape(Anoise_vec_hist, [ell, nr, n, n], order='F')
    Bnoise_hist = np.reshape(Bnoise_vec_hist, [ell, nr, n, m], order='F')

    return u_mean_hist,u_covr_hist,u_hist,Anoise_hist,Bnoise_hist


def estimate_model(n,m,nr,ell,x_hist,u_mean_hist,u_covr_hist,display_estimates=False,cheat_AB=False):
    muhat_hist = np.zeros([ell + 1, n])
    Xhat_hist = np.zeros([ell + 1, n * n])
    What_hist = np.zeros([ell + 1, n * m])

    # First stage: mean dynamics parameter estimation
    # Form data matrices for least-squares estimation
    for t in range(ell + 1):
        muhat_hist[t] = (1 / nr) * np.sum(x_hist[t], axis=0)
        Xhat_hist[t] = (1 / nr) * vec(np.sum(np.einsum('...i,...j', x_hist[t], x_hist[t]), axis=0))
        if t < ell:
            # What_hist[t] = (1/nr)*vec(np.sum(np.einsum('...i,...j',x_hist[t],u_mean_hist[t]),axis=0))
            What_hist[t] = vec(np.outer(muhat_hist[t], u_mean_hist[t]))
    Y = muhat_hist[1:].T
    Z = np.vstack([muhat_hist[0:-1].T, u_mean_hist.T])
    # Solve least-squares problem
    Thetahat = mdot(Y, Z.T, la.pinv(mdot(Z, Z.T)))
    # Split learned model parameters
    Ahat = Thetahat[:, 0:n]
    Bhat = Thetahat[:, n:n + m]

    if cheat_AB:
        Ahat = np.copy(A)
        Bhat = np.copy(B)

    AAhat = np.kron(Ahat, Ahat)
    ABhat = np.kron(Ahat, Bhat)
    BAhat = np.kron(Bhat, Ahat)
    BBhat = np.kron(Bhat, Bhat)

    # Second stage: covariance dynamics parameter estimation
    # Form data matrices for least-squares estimation
    C = np.zeros([ell, n * n]).T
    Uhat_hist = np.zeros([ell, m * m])
    for t in range(ell):
        Uhat_hist[t] = vec(u_covr_hist[t] + np.outer(u_mean_hist[t], u_mean_hist[t]))
        Cminus = mdot(AAhat,Xhat_hist[t])+mdot(BAhat,What_hist[t])+mdot(ABhat,What_hist[t].T)+mdot(BBhat,Uhat_hist[t])
        C[:, t] = Xhat_hist[t + 1] - Cminus
    D = np.vstack([Xhat_hist[0:-1].T, Uhat_hist.T])
    # Solve least-squares problem
    SigmaThetahat_prime = mdot(C, D.T, la.pinv(mdot(D, D.T)))
    # Split learned model parameters
    SigmaAhat_prime = SigmaThetahat_prime[:, 0:n * n]
    SigmaBhat_prime = SigmaThetahat_prime[:, n * n:n * (n + m)]

    # Reshape and project the noise covariance estimates onto the semidefinite cone
    SigmaAhat = reshaper(SigmaAhat_prime, n, n, n, n)
    SigmaBhat = reshaper(SigmaBhat_prime, n, m, n, m)
    SigmaAhat = positive_semidefinite_part(SigmaAhat)
    SigmaBhat = positive_semidefinite_part(SigmaBhat)

    if display_estimates:
        prettyprint(Ahat, "Ahat")
        prettyprint(A, "A   ")
        prettyprint(Bhat, "Bhat")
        prettyprint(B, "B   ")
        prettyprint(SigmaAhat, "SigmaAhat")
        prettyprint(SigmaA, "SigmaA   ")
        prettyprint(SigmaBhat, "SigmaBhat")
        prettyprint(SigmaB, "SigmaB   ")

    return Ahat,Bhat,SigmaAhat,SigmaBhat



def experiment_fixed_rollout(n,m,A,B,SigmaA,SigmaB,nr,ell):
    u_mean_hist, u_covr_hist, u_hist, Anoise_hist, Bnoise_hist = generate_sample_data(n,m,SigmaA,SigmaB,nr,ell)

    # Collect rollout data
    x_hist = np.zeros([ell + 1, nr, n])
    # Initialize the state
    x_hist[0] = npr.randn(nr, n)
    estimate_stride = 1
    ns = round(ell / estimate_stride)
    t_hist = np.arange(ell + 1)

    for t in range(ell):
        # Transition the state
        x_hist[t + 1] = groupdot(A + Anoise_hist[t], x_hist[t]) + groupdot(B + Bnoise_hist[t], u_hist[t])

    Ahat, Bhat, SigmaAhat, SigmaBhat = estimate_model(n, m, nr, ell, x_hist, u_mean_hist, u_covr_hist)

    # Plotting
    plot_trajectories(nr,ell,t_hist,x_hist)
    plot_model_estimates(A,B,SigmaA,SigmaB,Ahat,Bhat,SigmaAhat,SigmaBhat)


def experiment_increasing_rollout_length(n,m,A,B,SigmaA,SigmaB,nr,ell):
    u_mean_hist, u_covr_hist, u_hist, Anoise_hist, Bnoise_hist = generate_sample_data(n, m, SigmaA, SigmaB, nr, ell)

    # Collect rollout data
    x_hist = np.zeros([ell + 1, nr, n])
    # Initialize the state
    x_hist[0] = npr.randn(nr, n)
    estimate_stride = 1
    ns = round(ell / estimate_stride)
    t_hist = np.arange(ell + 1)
    s_hist = estimate_stride * np.arange(ns)
    Ahat_error_hist = np.full(ns, np.nan)
    Bhat_error_hist = np.full(ns, np.nan)
    SigmaAhat_error_hist = np.full(ns, np.nan)
    SigmaBhat_error_hist = np.full(ns, np.nan)
    k = 0
    for t in range(ell):
        # Transition the state
        x_hist[t+1] = groupdot(A+Anoise_hist[t],x_hist[t]) + groupdot(B+Bnoise_hist[t],u_hist[t])

        if t % estimate_stride == 0:
            Ahat,Bhat,SigmaAhat,SigmaBhat = estimate_model(n,m,nr,t,x_hist[0:t+1],u_mean_hist[0:t],u_covr_hist[0:t])
            Ahat_error_hist[k] = la.norm(A-Ahat)/la.norm(A)
            Bhat_error_hist[k] = la.norm(B-Bhat)/la.norm(B)
            SigmaAhat_error_hist[k] = la.norm(SigmaA-SigmaAhat)/la.norm(SigmaA)
            SigmaBhat_error_hist[k] = la.norm(SigmaB-SigmaBhat)/la.norm(SigmaB)
            k += 1

    # Plotting
    plot_trajectories(nr,ell,t_hist,x_hist)
    plot_estimation_error(s_hist, Ahat_error_hist, Bhat_error_hist, SigmaAhat_error_hist, SigmaBhat_error_hist, xlabel_str="Time step")
    plot_model_estimates(A,B,SigmaA,SigmaB,Ahat,Bhat,SigmaAhat,SigmaBhat)
    plt.show()


def experiment_increasing_rollout_count(n,m,A,B,SigmaA,SigmaB,nr,ell,u_mean_var,u_covr_var,estimate_stride=1,print_updates=True):
    u_mean_hist, u_covr_hist, u_hist, Anoise_hist, Bnoise_hist = generate_sample_data(n,m,SigmaA,SigmaB,nr,ell,u_mean_var,u_covr_var)

    # Collect rollout data
    x_hist = np.zeros([ell + 1, nr, n])
    # Initialize the state
    x_hist[0] = npr.randn(nr, n)
    ns = round(nr / estimate_stride)
    t_hist = np.arange(ell + 1)
    Ahat_error_hist = np.full(ns, np.nan)
    Bhat_error_hist = np.full(ns, np.nan)
    SigmaAhat_error_hist = np.full(ns, np.nan)
    SigmaBhat_error_hist = np.full(ns, np.nan)

    for t in range(ell):
        # Transition the state
        x_hist[t + 1] = groupdot(A + Anoise_hist[t], x_hist[t]) + groupdot(B + Bnoise_hist[t], u_hist[t])
        if print_updates:
            print("Simulated time step %d" % t)

    k = 0
    if print_updates:
        header_str = "# of rollouts |   A error   |   B error   | SigmaA error | SigmaB error"
        print(header_str)
    for r in np.arange(1,nr):
        if r % estimate_stride == 0:
            Ahat, Bhat, SigmaAhat, SigmaBhat = estimate_model(n, m, r, ell, x_hist[:,0:r], u_mean_hist, u_covr_hist)
            Ahat_error_hist[k] = la.norm(A-Ahat)/la.norm(A)
            Bhat_error_hist[k] = la.norm(B-Bhat)/la.norm(B)
            SigmaAhat_error_hist[k] = la.norm(SigmaA-SigmaAhat)/la.norm(SigmaA)
            SigmaBhat_error_hist[k] = la.norm(SigmaB-SigmaBhat)/la.norm(SigmaB)
            if print_updates:
                update_str = "%13d    %.3e     %.3e      %.3e      %.3e" % (r,Ahat_error_hist[k],Bhat_error_hist[k],SigmaAhat_error_hist[k],SigmaBhat_error_hist[k])
                print(update_str)
            k += 1
    # Plotting
    # plot_trajectories(nr,ell,t_hist,x_hist)
    # plot_estimation_error(s_hist, Ahat_error_hist, Bhat_error_hist, SigmaAhat_error_hist, SigmaBhat_error_hist, xlabel_str="Number of rollouts")
    # plot_model_estimates(A,B,SigmaA,SigmaB,Ahat,Bhat,SigmaAhat,SigmaBhat)
    # plt.show()

    return np.vstack([Ahat_error_hist, Bhat_error_hist, SigmaAhat_error_hist, SigmaBhat_error_hist])


def multi_experiment_increasing_rollout_count(n,m,A,B,SigmaA,SigmaB,ne,estimate_stride,ns,s_hist,nr,ell,u_mean_var,u_covr_var):
    experiment_data = np.zeros([4, ns, ne])
    for i in range(ne):
        experiment_data[:,:,i] = experiment_increasing_rollout_count(n,m,A,B,SigmaA,SigmaB,nr,ell,u_mean_var,u_covr_var,estimate_stride=estimate_stride,print_updates=False)
        print("Experiment %04d / %04d completed" % (i+1, ne))
    plot_estimation_error_multi(s_hist, experiment_data, xlabel_str="Number of rollouts")




def parameter_study():
    # Scalar experiment
    seed = 0
    npr.seed(seed)


    # A_noise_list =
    # B_noise_list =
    # u_mean_parameter_list =
    # u_covr_parameter_list =
    #
    # cheat_AB =

    # Number of rollouts
    nr = int(1e6)

    # Rollout length
    ell = 4

    # Model estimation stride
    estimate_stride = int(np.ceil(nr/100))
    ns = round(nr/estimate_stride)
    s_hist = estimate_stride*np.arange(ns)

    # Number of experiments
    ne = 40

    # System definition
    Sa = 0.3
    Sb = 0.5
    n,m,A,B,SigmaA,SigmaB = example_system_scalar(Sa,Sb)

    # Input design hyperparameters
    u_mean_var = 1
    u_covr_var = 0.1

    multi_experiment_increasing_rollout_count(n,m,A,B,SigmaA,SigmaB,ne,estimate_stride,ns,s_hist,nr,ell,u_mean_var,u_covr_var)


########################################################################################################################
# Main
########################################################################################################################

# for development only
import matplotlib.pyplot as plt
plt.close('all')

parameter_study()

# # Two-state experiment
# seed = 0
# npr.seed(seed)
# n,m,A,B,SigmaA,SigmaB = example_system_twostate()
# # Number of rollouts
# nr = int(1e6)
# # Rollout length
# ell = int((m**2*n**4)/2 + (m**2*n**2)/2 + m**2 + 1)
# experiment_increasing_rollout_count(n,m,A,B,SigmaA,SigmaB,nr,ell)


# Other experiements
# n,m,A,B,SigmaA,SigmaB = random_system(n=4,m=3,seed=seed)

# experiment_fixed_rollout(n,m,A,B,SigmaA,SigmaB,nr,ell)
# experiment_increasing_rollout_length(n,m,A,B,SigmaA,SigmaB,nr,ell)