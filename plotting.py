import numpy as np
import numpy.random as npr
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


def plot_trajectories(nr,ell,t_hist,x_hist):
    # Plot the rollout state data
    if ell < 1200 and nr < 4000:
        fig, ax = plt.subplots(n)
        plot_alpha = np.min([1, 10 / nr])
        if n > 1:
            for i in range(n):
                ax[i].step(t_hist, x_hist[:, :, i], color='tab:blue', linewidth=0.5, alpha=plot_alpha)
                ax[i].set_ylabel("State %d" % i+1)
            ax[-1].set_xlabel("Time step")
            ax[0].set_title("Rollout data")
        else:
            ax.step(t_hist, x_hist[:, :, 0], color='tab:blue', linewidth=0.5, alpha=plot_alpha)
            ax.set_ylabel("State")
            ax.set_xlabel("Time step")
            ax.set_title("Rollout data")
        return fig,ax


def plot_model_estimates(A,B,SigmaA,SigmaB,Ahat,Bhat,SigmaAhat,SigmaBhat):
    if A.size + B.size > 2:
        # View the model estimates as matrices
        fig, ax = plt.subplots(3, 4)
        fig.set_size_inches(10, 6)
        im00 = ax[0, 0].imshow(A)
        ax[0, 1].imshow(B)
        ax[0, 2].imshow(SigmaA)
        ax[0, 3].imshow(SigmaB)
        im10 = ax[1, 0].imshow(Ahat)
        ax[1, 1].imshow(Bhat)
        ax[1, 2].imshow(SigmaAhat)
        ax[1, 3].imshow(SigmaBhat)
        im20 = ax[2, 0].imshow(np.abs(A - Ahat))
        ax[2, 1].imshow(np.abs(B - Bhat))
        ax[2, 2].imshow(np.abs(SigmaA - SigmaAhat))
        ax[2, 3].imshow(np.abs(SigmaB - SigmaBhat))

        ax[0, 0].set_ylabel("True")
        ax[1, 0].set_ylabel("Estimate")
        ax[2, 0].set_ylabel("Normalized Error")
        ax[2, 0].set_xlabel("A")
        ax[2, 1].set_xlabel("B")
        ax[2, 2].set_xlabel("SigmaA")
        ax[2, 3].set_xlabel("SigmaB")

        # plt.colorbar(im00,ax=ax[0,3])
        # plt.colorbar(im10,ax=ax[1,3])
        # plt.colorbar(im20,ax=ax[2,3])
        return fig, ax


def plot_estimation_error(tk_hist,Ahat_error_hist,Bhat_error_hist,SigmaAhat_error_hist,SigmaBhat_error_hist,xlabel_str):
    # Plot the normalized model estimation errors
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 6)
    ax.step(tk_hist, Ahat_error_hist, linewidth=2)
    ax.step(tk_hist, Bhat_error_hist, linewidth=2)
    ax.step(tk_hist, SigmaAhat_error_hist, linewidth=2)
    ax.step(tk_hist, SigmaBhat_error_hist, linewidth=2)
    ax.legend(["Ahat", "Bhat", "SigmaAhat", "SigmaBhat"])
    ax.set_xlabel(xlabel_str)
    ax.set_ylabel("Normalized Error")
    # ax.set_yscale("log")
    return fig,ax


def plot_estimation_error_multi(s_hist,experiment_data,xlabel_str,scale_option='log'):
    # Plot the normalized model estimation errors
    fig,ax = plt.subplots(nrows=2,ncols=2)
    plt.subplots_adjust(wspace=0.3,hspace=0.3)
    fig.set_size_inches(8, 8)
    # title_str_list = ["A", "B", "SigmaA", "SigmaB"]
    title_str_list = ["A", "B", r"$\Sigma_A$", r"$\Sigma_B$"]
    ax_idx_i = [0,0,1,1]
    ax_idx_j = [0,1,0,1]
    for k in range(4):
        i,j = ax_idx_i[k],ax_idx_j[k]
        # Fill the region between min and max values
        ax[i,j].fill_between(s_hist,np.min(experiment_data[k],1),np.max(experiment_data[k],1), step='pre', color='silver', alpha = 0.5)
        # Fill the interquartile region
        ax[i,j].fill_between(s_hist,np.percentile(experiment_data[k],25,1),np.percentile(experiment_data[k],75,1), step='pre', color='grey',alpha=0.5)
        # Plot the individual experiment realizations
        ax[i,j].step(s_hist, experiment_data[k], linewidth=1, alpha=0.5)
        # ax[i,j].step(s_hist, experiment_data[k], color='tab:blue', linewidth=1, alpha=0.5)
        # # Plot the mean of the experiments
        # ax[i,j].step(s_hist, np.mean(experiment_data[k],1), color='mediumblue', linewidth=2)
        # Plot the median of the experiments
        ax[i,j].step(s_hist, np.percentile(experiment_data[k], 50, 1), color='k', linewidth=2)
        if scale_option == 'log':
            # Plot a reference curve for an O(1/sqrt(N)) convergence rate
            ref_scale = 1.0
            ref_curve = s_hist**-0.5
            ref_curve *= ref_scale*np.max(np.percentile(experiment_data[k],75,1)/ref_curve)
            ax[i,j].plot(s_hist, ref_curve, color='r', linewidth=2, linestyle='--')
            ax[i,j].set_xscale("log")
            ax[i,j].set_yscale("log")
        ax[i,j].set_title(title_str_list[k],fontsize=20)
        ax[i,j].set_xlabel(xlabel_str)
        ax[i,j].set_ylabel("Normalized Error")
        # ax[i,j].set_ylim([1e-4,1e1])
        if ax[i,j].get_xscale() is "linear":
            ax[i,j].ticklabel_format(axis='x',style='sci',scilimits=(0,4))
    return fig,ax