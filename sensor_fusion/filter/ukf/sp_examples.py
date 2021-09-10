import pdb
import numpy as np
from numpy.random import multivariate_normal
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
sns.set()

print("Import done.")


# ========== Propagate a sampled distribution
# create our nonlinear function
def some_nonlinear_function(x, y):
    return np.array([x + y, .1*x**2 + y**2])

# create our distribution
mean = (0.,0.)
cov = np.array([[32., 15.], [15., 40.]])
mean_prop_og = some_nonlinear_function(*mean)

# sample and propagate through the nonlinear function
num_sample = 10000
x_sample, y_sample = multivariate_normal(mean=mean, cov=cov, size=num_sample).T
xy_sample_prop = np.zeros((num_sample,2))
for idx in range(num_sample):
    xy_sample_prop[idx,:] = some_nonlinear_function(x_sample[idx], y_sample[idx])
mean_prop = np.mean(xy_sample_prop, axis=0)

# prepare data for plotting
data_plot = pd.DataFrame({"X":x_sample, "Y":y_sample})
data_plot_prop = pd.DataFrame({"X":xy_sample_prop[:,0], "Y":xy_sample_prop[:,1]})

# plot the propagated samples and the mean
fig, axes = plt.subplots(1,2)
axes[0].set_title("Before Propagation")
axes[1].set_title("After Propagation")
sns.scatterplot(ax=axes[0],x="X", y="Y", data=data_plot, alpha=0.05)
sns.scatterplot(ax=axes[1],x="X", y="Y", data=data_plot_prop, alpha=0.05)
axes[0].scatter(mean[0], mean[1], c=sns.color_palette()[1])
axes[1].scatter(mean_prop_og[0], mean_prop_og[1], c=sns.color_palette()[1])
axes[1].scatter(mean_prop[0], mean_prop[1], c=[1,1,0], marker="*")
plt.show()

print("Propagating sampled points done.")


# ========== Propagating sigma points
from scipy.linalg import cholesky

def generate_vandermerwe_sigma_point(mean, cov, n, alpha, beta, kappa):
    """Generate sigma points based on the Van der Merwe's Scalaed
    Sigma Point algorithm.

    Args:
        mean (np.ndarray): Mean
        cov (np.nadarray): Covariance
        n (int): Dimension of the state.
        alpha (float): Parameter
        beta (float): Parameter
        kappa (float): Parameter
    """
    lamb = (alpha**2)*(n+kappa)-n
    U = cholesky((lamb+n)*cov)

    # generate sigma points
    sigma = np.zeros((2*n+1,n))
    sigma[0,None,:] = mean.flatten()
    for idx in range(n):
        sigma[idx+1,:] = np.subtract(mean.flatten(), -U[idx])
        sigma[n+idx+1,:] = np.subtract(mean.flatten(), U[idx])

    # generate weights
    weights_mean = np.ones((2*n+1,1))
    weights_cov = np.ones((2*n+1,1))

    weights_mean *= 1/(2*(n+lamb))
    weights_cov *= 1/(2*(n+lamb))

    weights_mean[0] = lamb/(n+lamb)
    weights_cov[0] = (lamb/(n+lamb)) + 1 - alpha**2 + beta

    return (sigma, weights_mean, weights_cov)

def unscented_transform(sigma, w_m, w_c, Q=None):
    """Computes the unscented transform of the sigma points.

    Args:
        sigma (np.ndarray): Sigma points
        w_m (np.ndarray): Mean weights
        w_c (np.ndarray): Mean covariance
        Q (np.ndarray): Noise covariance
    """
    num_sigma = sigma.shape[0]
    num_state = sigma.shape[1]
    
    mean = np.dot(w_m.T, sigma)
    residual = sigma-mean
    cov = np.dot(residual.T, np.dot(np.diag(w_c[:,0]), residual))
    
    if isinstance(Q, np.ndarray):
        cov += Q
    return (mean.T, cov)


(sigma, w_mean, w_cov) = generate_vandermerwe_sigma_point(np.asarray(mean), cov, 2, .3, 2., .1)

num_sigma = sigma.shape[0]
xy_sigma_prop = np.zeros((num_sigma,2))
for idx in range(num_sigma):
    xy_sigma_prop[idx,:] = some_nonlinear_function(sigma[idx,0], sigma[idx,1])
mean_sigma_prop = np.mean(xy_sigma_prop, axis=0)

mean_unscented, cov_unscented = unscented_transform(xy_sigma_prop, w_mean, w_cov)

fig, axes = plt.subplots(1,2)
axes[0].set_title("Before Propagation Sigma")
axes[1].set_title("After Propagation Sigma")
sns.scatterplot(ax=axes[0],x="X", y="Y", data=data_plot, alpha=0.05)
sns.scatterplot(ax=axes[1],x="X", y="Y", data=data_plot_prop, alpha=0.05)
axes[0].scatter(sigma[:,0], sigma[:,1], c=sns.color_palette()[1])
axes[1].scatter(mean_unscented[0], mean_unscented[1], c=sns.color_palette()[1])
axes[1].scatter(mean_prop[0], mean_prop[1], c=[1,1,0], marker="*")
plt.show()

print("Propagating sigma points done.")


# ========== Sigma point interpretation
alpha = 1.3
beta = 2.
kappa = .1

(sigma, w_mean, w_cov) = generate_vandermerwe_sigma_point(np.asarray(mean), cov, 2, alpha=alpha, beta=beta, kappa=kappa)

num_sigma = sigma.shape[0]
xy_sigma_prop = np.zeros((num_sigma,2))
for idx in range(num_sigma):
    xy_sigma_prop[idx,:] = some_nonlinear_function(sigma[idx,0], sigma[idx,1])
mean_sigma_prop = np.mean(xy_sigma_prop, axis=0)

mean_unscented, cov_unscented = unscented_transform(xy_sigma_prop, w_mean, w_cov)

fig, axes = plt.subplots(1,2)
axes[0].set_title("Before Propagation Sigma")
axes[1].set_title("After Propagation Sigma")
sns.scatterplot(ax=axes[0],x="X", y="Y", data=data_plot, alpha=0.05)
sns.scatterplot(ax=axes[1],x="X", y="Y", data=data_plot_prop, alpha=0.05)
for idx in range(sigma.shape[0]):
    axes[0].scatter(sigma[idx,0], sigma[idx,1], c=sns.color_palette()[1])#, s=500*abs(w_mean[idx,0]))
# axes[0].scatter(sigma[:,0], sigma[:,1])
axes[1].scatter(mean_unscented[0], mean_unscented[1], c=sns.color_palette()[1])
axes[1].scatter(mean_prop[0], mean_prop[1], c=[1,1,0], marker="*")
plt.show()

print(w_mean.T)
print(w_cov.T)

print(cov_unscented)