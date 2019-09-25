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


def plot_estimation_error_multi(n,m,s_hist,experiment_data,xlabel_str,scale_option='log',show_reference_curve=True):
    # Plot the normalized model estimation errors
    fig,ax = plt.subplots(nrows=2,ncols=2)
    plt.subplots_adjust(wspace=0.4,hspace=0.4)
    fig.set_size_inches(7, 7)
    # title_str_list = ["A", "B", "SigmaA", "SigmaB"]
    title_str_list = ["A", "B", r"$\Sigma_A$", r"$\Sigma_B$"]
    # ylabel_str_list = [r"$ \frac{\|A-\hat{A}\|_F}{\|A\|_F} $",
    #                    r"$ \frac{\|B-\hat{B}\|_F}{\|B\|_F} $",
    #                    r"$ \frac{\|\Sigma_A-\hat{\Sigma}_A \|_F}{\|\Sigma_A\|_F} $",
    #                    r"$ \frac{\|\Sigma_B-\hat{\Sigma}_B \|_F}{\|\Sigma_B\|_F} $"]
    ylabel_str_list = ["Normalized error"]*4
    ax_idx_i = [0,0,1,1]
    ax_idx_j = [0,1,0,1]
    for k in range(4):
        # Get quartiles
        y000 = np.min(experiment_data[k],1)
        y025 = np.percentile(experiment_data[k],25,1)
        y050 = np.percentile(experiment_data[k],50,1)
        y075 = np.percentile(experiment_data[k],75,1)
        y100 = np.max(experiment_data[k],1)
        # Axes indices
        i,j = ax_idx_i[k], ax_idx_j[k]
        # Fill the region between min and max values
        ax[i,j].fill_between(s_hist,y000,y100,step='pre',color=0.6*np.ones(3),alpha=0.5)
        # Fill the interquartile region
        ax[i,j].fill_between(s_hist,y025,y075,step='pre',color=0.3*np.ones(3),alpha=0.5)
        # Plot the individual experiment realizations
        if experiment_data.shape[2] < 8:
            ax[i,j].step(s_hist, experiment_data[k], linewidth=1, alpha=0.6)
        else:
            # ax[i,j].step(s_hist, experiment_data[k], color='tab:blue', linewidth=1, alpha=0.2)
            pass
        # # Plot the mean of the experiments
        # ax[i,j].step(s_hist, np.mean(experiment_data[k],1), color='mediumblue', linewidth=2)
        # Plot the median of the experiments
        median_handle, = ax[i,j].step(s_hist,y050,color='k',linewidth=2)
        # Plot a reference curve for an O(1/sqrt(N)^n or m) convergence rate
        if show_reference_curve:
            ref_scale = 1.5
            exponent = 1 if i==0 else n if j==0 else m
            ref_curve = s_hist**(-0.5/exponent)
            ref_curve *= ref_scale*np.max(np.percentile(experiment_data[k],75,1)/ref_curve)
            ref_handle, = ax[i,j].plot(s_hist, ref_curve, color='r', linewidth=2, linestyle='--')
        if scale_option == 'log':
            ax[i,j].set_xscale("log")
            ax[i,j].set_yscale("log")
        ax[i,j].set_title(title_str_list[k],fontsize=16)
        ax[i,j].set_xlabel(xlabel_str,fontsize=12)
        ax[i,j].set_ylabel(ylabel_str_list[k],rotation=90,fontsize=12)
        # ax[i,j].set_ylim([1e-4,1e1])
        if ax[i,j].get_xscale() is "linear":
            ax[i,j].ticklabel_format(axis='x',style='sci',scilimits=(0,4))
        legend_handles = (median_handle,ref_handle)
        legend_labels = (r"Median",r"$\mathcal{O}\left(n_r^{-1/%d} \right)$" % (2*exponent))
        ax[i,j].legend(legend_handles,legend_labels,fontsize=12)
        xtick_max = np.log10(s_hist.max())
        xtick_vals = np.logspace(0,xtick_max,int(xtick_max)+1)
        ax[i,j].set_xticks(xtick_vals)
    return fig,ax