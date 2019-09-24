import numpy as np
import numpy.random as npr
import numpy.linalg as la

from matrixmath import specrad, mdot, vec, sympart, positive_semidefinite_part

from time import time

from system_definitions import random_system, example_system_scalar, example_system_twostate

from system_identification import generate_sample_data, collect_rollouts, estimate_model

from plotting import plot_trajectories,plot_model_estimates,plot_estimation_error,plot_estimation_error_multi


########################################################################################################################
# Functions
########################################################################################################################

def groupdot(A,x):
    return np.einsum('...jk,...k',A,x)


def experiment_fixed_rollout(n,m,A,B,SigmaA,SigmaB,nr,ell):
    # Generate sample data
    u_mean_hist, u_covr_hist, u_hist, Anoise_hist, Bnoise_hist = generate_sample_data(n,m,SigmaA,SigmaB,nr,ell)

    # Collect rollout data
    x_hist = collect_rollouts(n,m,A,B,nr,ell,Anoise_hist,Bnoise_hist,u_hist)
    t_hist = np.arange(ell+1)

    # Estimate the model
    Ahat, Bhat, SigmaAhat, SigmaBhat = estimate_model(n,m,A,B,SigmaA,SigmaB,nr,ell,x_hist,u_mean_hist,u_covr_hist)

    # Plotting
    plot_trajectories(nr,ell,t_hist,x_hist)
    plot_model_estimates(A,B,SigmaA,SigmaB,Ahat,Bhat,SigmaAhat,SigmaBhat)


def experiment_increasing_rollout_length(n,m,A,B,SigmaA,SigmaB,nr,ell):
    # Generate sample data
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
            Ahat,Bhat,SigmaAhat,SigmaBhat = estimate_model(n,m,A,B,SigmaA,SigmaB,nr,t,x_hist[0:t+1],u_mean_hist[0:t],u_covr_hist[0:t])
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


def experiment_increasing_rollout_count(n,m,A,B,SigmaA,SigmaB,nr,ell,u_mean_var,u_covr_var,ns,s_hist,print_updates=True):
    u_mean_hist,u_covr_hist,u_hist,Anoise_hist,Bnoise_hist = generate_sample_data(n,m,SigmaA,SigmaB,nr,ell,u_mean_var,u_covr_var)

    # Collect rollout data
    x_hist = collect_rollouts(n,m,A,B,nr,ell,Anoise_hist,Bnoise_hist,u_hist)
    t_hist = np.arange(ell+1)

    # Estimate the model for increasing numbers of rollouts
    Ahat_error_hist = np.full(ns,np.nan)
    Bhat_error_hist = np.full(ns,np.nan)
    SigmaAhat_error_hist = np.full(ns,np.nan)
    SigmaBhat_error_hist = np.full(ns,np.nan)
    k = 0
    if print_updates:
        header_str = "# of rollouts |   A error   |   B error   | SigmaA error | SigmaB error"
        print(header_str)
    for r in np.arange(1,nr+1):
        if r == s_hist[k]:
            Ahat,Bhat,SigmaAhat,SigmaBhat = estimate_model(n,m,A,B,SigmaA,SigmaB,r,ell,x_hist[:,0:r],u_mean_hist,u_covr_hist)
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


def multi_experiment_increasing_rollout_count(n,m,A,B,SigmaA,SigmaB,ne,ns,s_hist,nr,ell,u_mean_var,u_covr_var):
    experiment_data = np.zeros([4,ns,ne])
    for i in range(ne):
        experiment_data[:,:,i] = experiment_increasing_rollout_count(n,m,A,B,SigmaA,SigmaB,nr,ell,u_mean_var,u_covr_var,ns,s_hist,print_updates=True)
        print("Experiment %04d / %04d completed" % (i+1,ne))
    plot_estimation_error_multi(s_hist, experiment_data, xlabel_str="Number of rollouts")


def parameter_study(variable_parameter,variable_parameter_list,nominal_parameters):
    # Scalar experiment
    seed = 1
    npr.seed(seed)

    # Number of rollouts
    nr = int(1e4)

    # Rollout length
    ell = 4

    # Model estimation points
    estimation_points = 'log'
    ns = 100
    if estimation_points == 'linear':
        s_hist = np.round(np.linspace(0, nr, ns + 1)).astype(int)[1:]
    elif estimation_points == 'log':
        s_hist = np.round(np.logspace(0,np.log10(nr),ns+1,base=10)).astype(int)[1:]
        s_hist = np.unique(s_hist)
        ns = s_hist.size

    # Number of experiments
    ne = 20

    # System definition
    Sa = nominal_parameters["Sa"]
    Sb = nominal_parameters["Sb"]

    # Input design hyperparameters
    u_mean_var = nominal_parameters["u_mean_std"]
    u_covr_var = nominal_parameters["u_covr_std"]

    for parameter_val in variable_parameter_list:
        if variable_parameter == "Sa":
            Sa = parameter_val
        elif variable_parameter == "Sb":
            Sb = parameter_val
        elif variable_parameter == "u_mean_std":
            u_mean_var = parameter_val
        elif variable_parameter == "u_covr_std":
            u_covr_var = parameter_val
        n,m,A,B,SigmaA,SigmaB = example_system_scalar(Sa,Sb)
        multi_experiment_increasing_rollout_count(n,m,A,B,SigmaA,SigmaB,ne,ns,s_hist,nr,ell,u_mean_var,u_covr_var)


def multiple_parameter_study():
    nominal_parameters = {"Sa": 0.5, "Sb": 0.5, "u_mean_std": 1.0, "u_covr_std": 0.1}

    parameter_list = {"Sa": np.logspace(-3, 1, 3, base=10),
                      "Sb": np.logspace(-3, 1, 3, base=10),
                      "u_mean_std": np.logspace(-3, 1, 3, base=10),
                      "u_covr_std": np.logspace(-3, 1, 3, base=10)}

    for key in nominal_parameters.keys():
        print("----Performing parameter study on %s ----" % key)
        parameter_study(key, parameter_list[key], nominal_parameters)
        print('')


########################################################################################################################
# Main
########################################################################################################################

# for development only
import matplotlib.pyplot as plt
plt.close('all')


# Scalar experiment
seed = 2
npr.seed(seed)

# System definition
# n,m,A,B,SigmaA,SigmaB = example_system_scalar()
# n,m,A,B,SigmaA,SigmaB = example_system_twostate()
n,m,A,B,SigmaA,SigmaB = random_system(n=2,m=2,seed=seed)

# Number of rollouts
nr = int(1e4)

# Rollout length
ell = int((m**2*n**4)/2 + (m**2*n**2)/2 + m**2 + 1)

# Model estimation points
estimation_points = 'log'
ns = 100
if estimation_points == 'linear':
    s_hist = np.round(np.linspace(0, nr, ns + 1)).astype(int)[1:]
elif estimation_points == 'log':
    s_hist = np.round(np.logspace(0,np.log10(nr),ns+1,base=10)).astype(int)[1:]
    s_hist = np.unique(s_hist)
    ns = s_hist.size

# Number of experiments
ne = 4

# u_mean_std = 1
# u_covr_std = 0.1
u_mean_std = la.norm(SigmaB,2)
u_covr_std = u_mean_std/10

multi_experiment_increasing_rollout_count(n, m, A, B, SigmaA, SigmaB, ne, ns, s_hist, nr, ell, u_mean_std, u_covr_std)

# experiment_increasing_rollout_count(n,m,A,B,SigmaA,SigmaB,nr,ell,u_mean_std,u_covr_std,ns,s_hist,print_updates=True)



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