"""

This program runs experiments to generate the plots in the paper. 

"""

import pylab
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import tensorly.random
import tensorly as tl
import h5py
import mat73

from decomp import *

"""

1. Import real world tensor if using one. Skip to step 2 if using a simulated 
tensor. 

Example of loading claus tensor from local repository is given below. 

"""

import scipy.io
#mat = scipy.io.loadmat('claus.mat', squeeze_me=True)
#tensor = mat['X']

#mat = scipy.io.loadmat('gas3data.mat', squeeze_me=True)
#tensor = mat['gas3_data']

#mat = mat73.loadmat('Wine_v7.mat')
#tensor = mat['Data_GC']

#mat = scipy.io.loadmat('fia.mat', squeeze_me=True)
#tensor = mat['X']
#tensor = tensor.reshape(12, 100, 89)

"""

2. If using a simulated tensor, set its parameters. 

"""

# A. Set parameters for the simulated 3-way tensors. See 'simulate_tensors2' 
# function for details. 

size = 50 # Length of each dimension
l1 = 0.01 # Homoscedastic noise param
l2 = 0 # Heteroscedastic noise param
bottleneck = 0 
rank = 15 # Tensor's rank
noise=False # Whether the tensor contains noise
# Collinearity parameters:
a = [0.9,0.9,0.9]
b = [0.9,0.9,0.9]

"""

3. Set the parameters for the experiments. To generate a convergence plot, set 
fit_thresh > 1 and first_100_plot = True. To generate the acceleration ratios, 
set fit_thresh as 0.01 less than the optimal fit and first_100_plot = False. 

"""

fit_thresh = 0.99 # The threshold fit for the algorithm to have 'converged'. Set as greater than 1 for convergence plots.
generate_plot = True
if generate_plot:
    fit_thresh = 1.1
#fit_thresh = 0.98
nruns = 20 # Set number of runs (number of tensor decompositions for each method)
simulated_tensor = True # Set as True if using simulated tensor; False if using a real tensor
first_100_plot = True # Set as True to run each alg for n_iter_max iterations and produce the convergence plots


"""

4. Set parameters for the algorithms. 

"""

decomp_rank = rank # Rank of the decomposition
n_iter_max = 600 # Maximum number of iterations to run algorithm for
tol=10e-6000000 # Error tolerance for termination
thresh = 0.99 # Collinearity threshold for BA-ALS and BP-ALS
noise_prop = 0.5

"""

5. Run repeated experiments: apply each tensor decomposition algorithm to 
the given tensor (real world) or tensor type (simulated). 

"""


"""

(i) ALS blockwise momentum.

"""

# Storage:
final_errors_block_momentum = []
final_iters_block_momentum = []
final_times_block_momentum = []
final_fits_block_momentum = []
fits_block_momentum = np.zeros([(n_iter_max-1),nruns])
times_block_momentum = np.zeros([(n_iter_max-1),nruns])

for j in range(nruns):
    
    print("BM-ALS", j)
    
    if simulated_tensor == True:
        Z, Zprime, Zdprime, U = simulate_tensors2(random_seed=j, I=size, a=a, b=b, R=rank, l1=l1, l2=l2, bottleneck=bottleneck, torch_tensor=False)
        tensor=Zdprime
    elif simulated_tensor == False:
        tensor = np.asarray(tensor)
    factors, final_error, elapsed_time, total_iters, errors, times, best_errors = als_blockwise(tensor=tensor, rank=decomp_rank, random_state=j, method='momentum', tol=tol, n_iter_max=n_iter_max, thresh=thresh, noise_prop=noise_prop, weight=0.5, fitness_threshold=fit_thresh)
    if first_100_plot==True:
        #fits_block_momentum[:,j] = [1 - x for x in errors/tl.norm(tensor)]
        fits_block_momentum[:,j] = [1 - x for x in best_errors/tl.norm(tensor)]
        times_block_momentum[:,j] = times
    if noise == True:
        tl.set_backend('numpy')
        final_error = tl.norm(tensor - tl.kruskal_to_tensor((np.ones(rank), factors)))
        final_errors_block_momentum.append(final_error)
    else: 
        final_errors_block_momentum.append(final_error)
    final_fits_block_momentum.append(1-final_error/tl.norm(tensor))
    final_iters_block_momentum.append(total_iters)
    final_times_block_momentum.append(elapsed_time)
    
    print(final_times_block_momentum)


"""
(ii) ALS blockwise perturbation.

"""


# Storage:
final_errors_block_perturb = []
final_iters_block_perturb = []
final_times_block_perturb = []
final_fits_block_perturb = []
fits_block_perturb = np.zeros([(n_iter_max-1),nruns])
times_block_perturb = np.zeros([(n_iter_max-1),nruns])

for j in range(nruns):
    
    print("BP-ALS", j)
    
    if simulated_tensor==True:
        Z, Zprime, Zdprime, U = simulate_tensors2(random_seed=j, I=size, a=a, b=b, R=rank, l1=l1, l2=l2, bottleneck=bottleneck, torch_tensor=False)
        tensor=Zdprime
    elif simulated_tensor == False:
        tensor = np.asarray(tensor)
    factors, final_error, elapsed_time, total_iters, errors, times, best_errors = als_blockwise(tensor=tensor, rank=decomp_rank, random_state=j, method='perturb', tol=tol, n_iter_max=n_iter_max, thresh=thresh, noise_prop=noise_prop, weight=0.5, fitness_threshold=fit_thresh)
    if first_100_plot==True:
        #fits_block_perturb[:,j] = [1 - x for x in errors/tl.norm(tensor)]
        fits_block_perturb[:,j] = [1 - x for x in best_errors/tl.norm(tensor)]
        times_block_perturb[:,j] = times
    if noise == True:
        tl.set_backend('numpy')
        final_error = tl.norm(tensor - tl.kruskal_to_tensor((np.ones(rank), factors)))
        final_errors_block_perturb.append(final_error)
    else: 
        final_errors_block_perturb.append(final_error)
    final_fits_block_perturb.append(1-final_error/tl.norm(tensor))
    final_iters_block_perturb.append(total_iters)
    final_times_block_perturb.append(elapsed_time)


"""
(iii) BA-ALS.

"""

# Storage:
final_errors_block_both = []
final_iters_block_both = []
final_times_block_both = []
final_fits_block_both = []
fits_block_both = np.zeros([(n_iter_max-1),nruns])
times_block_both = np.zeros([(n_iter_max-1),nruns])

for j in range(nruns):
    
    print("BA-ALS", j)
    
    if simulated_tensor==True:
        Z, Zprime, Zdprime, U = simulate_tensors2(random_seed=j, I=size, a=a ,b=b, R=rank, l1=l1, l2=l2, bottleneck=bottleneck, torch_tensor=False)
        tensor=Zdprime
    elif simulated_tensor == False:
        tensor = np.asarray(tensor)
    factors, final_error, elapsed_time, total_iters, errors, times, best_errors = als_blockwise(tensor=tensor, rank=decomp_rank, random_state=j, method='momentum_and_perturb', tol=tol, n_iter_max=n_iter_max, thresh=thresh, noise_prop=noise_prop, weight=0.5, fitness_threshold=fit_thresh)
    if first_100_plot==True:
        #fits_block_both[:,j] = [1 - x for x in errors/tl.norm(tensor)]
        fits_block_both[:,j] = [1 - x for x in best_errors/tl.norm(tensor)]
        times_block_both[:,j] = times
    if noise == True:
        tl.set_backend('numpy')
        final_error = tl.norm(tensor - tl.kruskal_to_tensor((np.ones(rank), factors)))
        final_errors_block_both.append(final_error)
    else: 
        final_errors_block_both.append(final_error)
    final_fits_block_both.append(1-final_error/tl.norm(tensor))
    final_iters_block_both.append(total_iters)
    final_times_block_both.append(elapsed_time)

"""
(iv) ALS-momentum (Mitchell et al 2018).

"""

# Storage:
final_errors_s1 = []
final_iters_s1 = []
final_times_s1 = []
final_fits_s1 = []
fits_s1 = np.zeros([(n_iter_max-1),nruns])
times_s1 = np.zeros([(n_iter_max-1),nruns])

for j in range(nruns):
    
    print("M-ALS", j)
    
    if simulated_tensor==True:
        Z, Zprime, Zdprime, U = simulate_tensors2(random_seed=j, I=size, a=a, b=b, R=rank, l1=l1, l2=l2, bottleneck=bottleneck, torch_tensor=False)
        tensor=Zdprime
    elif simulated_tensor == False:
        tensor = np.asarray(tensor)
        
    factors, final_error, elapsed_time, total_iters, errors, times, best_errors = als_blockwise(tensor=tensor, rank=decomp_rank, random_state=j, method='ALS_momentum', tol=tol, n_iter_max=n_iter_max, thresh=thresh, noise_prop=noise_prop, weight=0.5, fitness_threshold=fit_thresh)
    if first_100_plot==True:
        #fits_s1[:,j] = 1-torch.tensor(errors)/tl.norm(tensor)
        fits_s1[:,j] = [1 - x for x in best_errors/tl.norm(tensor)]
        times_s1[:,j] = times
    if noise == True:
        final_error = tl.norm(tensor - tl.kruskal_to_tensor((np.ones(rank), factors)))
        final_errors_s1.append(final_error)
    else: 
        final_error = errors[-1]
        final_errors_s1.append(final_error)
    final_fits_s1.append(1-final_error/tl.norm(tensor))
    final_iters_s1.append(total_iters)
    final_times_s1.append(elapsed_time)


"""
(v) Standard ALS.

"""

# Storage:
final_errors_als = []
final_iters_als = []
final_times_als = []
final_fits_als = []
fits_als = np.zeros([(n_iter_max-1),nruns])
times_als = np.zeros([(n_iter_max-1),nruns])

for j in range(nruns):
    
    print("ALS", j)
    
    if simulated_tensor==True:
        Z, Zprime, Zdprime, U = simulate_tensors2(random_seed=j, I=size, a=a, b=b, R=rank, l1=l1, l2=l1, bottleneck=bottleneck, torch_tensor=False)
        tensor=Zdprime
    elif simulated_tensor == False:
        tensor = np.asarray(tensor)
    factors, final_error, elapsed_time, total_iters, errors, times, best_errors = als_blockwise(tensor=tensor, rank=decomp_rank, random_state=j, method='ALS', tol=tol, n_iter_max=n_iter_max, thresh=thresh, noise_prop=noise_prop, weight=0.5, fitness_threshold=fit_thresh)
    if first_100_plot==True:
        #fits_als[:,j] = 1-torch.tensor(errors)/tl.norm(tensor)
        fits_als[:,j] = [1 - x for x in best_errors/tl.norm(tensor)]
        times_als[:,j] = times
    if noise == True:
        final_error = tl.norm(tensor - tl.kruskal_to_tensor((np.ones(rank), factors)))
        final_errors_als.append(final_error)
    else: 
        final_error = errors[-1]
        final_errors_als.append(final_error)
    final_fits_als.append(1-final_error/tl.norm(tensor))
    final_iters_als.append(total_iters)
    final_times_als.append(elapsed_time)


"""
(vi) herALS.

"""

# Storage:
final_errors_her = []
final_iters_her = []
final_times_her = []
final_fits_her = []
fits_her = np.zeros([(n_iter_max-1),nruns])
times_her = np.zeros([(n_iter_max-1),nruns])

for j in range(nruns):
    
    print("herALS", j)
    
    if simulated_tensor==True:
        Z, Zprime, Zdprime, U = simulate_tensors2(random_seed=j, I=size, a=a, b=b, R=rank, l1=l1, l2=l1, bottleneck=bottleneck, torch_tensor=False)
        tensor=Zdprime
    elif simulated_tensor == False:
        tensor = np.asarray(tensor)
    factors, final_error, elapsed_time, total_iters, errors, times, best_errors = als_blockwise(tensor=tensor, rank=decomp_rank, random_state=j, method='herALS', tol=tol, n_iter_max=n_iter_max, thresh=thresh, noise_prop=noise_prop, weight=0.5, fitness_threshold=fit_thresh)
    if first_100_plot==True:
        #fits_als[:,j] = 1-torch.tensor(errors)/tl.norm(tensor)
        fits_her[:,j] = [1 - x for x in best_errors/tl.norm(tensor)]
        times_her[:,j] = times
    if noise == True:
        final_error = tl.norm(tensor - tl.kruskal_to_tensor((np.ones(rank), factors)))
        final_errors_her.append(final_error)
    else: 
        final_error = errors[-1]
        final_errors_her.append(final_error)
    final_fits_her.append(1-final_error/tl.norm(tensor))
    final_iters_her.append(total_iters)
    final_times_her.append(elapsed_time)


"""
(vi) LS.

"""

# Storage:
final_errors_ls = []
final_iters_ls = []
final_times_ls = []
final_fits_ls = []
fits_ls = np.zeros([(n_iter_max-1),nruns])
times_ls = np.zeros([(n_iter_max-1),nruns])

for j in range(nruns):
    
    print("LS", j)
    
    if simulated_tensor==True:
        Z, Zprime, Zdprime, U = simulate_tensors2(random_seed=j, I=size, a=a, b=b, R=rank, l1=l1, l2=l1, bottleneck=bottleneck, torch_tensor=False)
        tensor=Zdprime
    elif simulated_tensor == False:
        tensor = np.asarray(tensor)
    factors, final_error, elapsed_time, total_iters, errors, times, best_errors = als_blockwise(tensor=tensor, rank=decomp_rank, random_state=j, method='LS', tol=tol, n_iter_max=n_iter_max, thresh=thresh, noise_prop=noise_prop, weight=0.5, fitness_threshold=fit_thresh)
    if first_100_plot==True:
        #fits_als[:,j] = 1-torch.tensor(errors)/tl.norm(tensor)
        fits_ls[:,j] = [1 - x for x in best_errors/tl.norm(tensor)]
        times_ls[:,j] = times
    if noise == True:
        final_error = tl.norm(tensor - tl.kruskal_to_tensor((np.ones(rank), factors)))
        final_errors_ls.append(final_error)
    else: 
        final_error = errors[-1]
        final_errors_ls.append(final_error)
    final_fits_ls.append(1-final_error/tl.norm(tensor))
    final_iters_ls.append(total_iters)
    final_times_ls.append(elapsed_time)


"""
(vi) ELS.

"""

n_iter_max = 20

# Storage:
final_errors_els = []
final_iters_els = []
final_times_els = []
final_fits_els = []
fits_els = np.zeros([(n_iter_max-1),nruns])
times_els = np.zeros([(n_iter_max-1),nruns])

for j in range(nruns):
    
    print("ELS", j)
    
    if simulated_tensor==True:
        Z, Zprime, Zdprime, U = simulate_tensors2(random_seed=j, I=size, a=a, b=b, R=rank, l1=l1, l2=l1, bottleneck=bottleneck, torch_tensor=False)
        tensor=Zdprime
    elif simulated_tensor == False:
        tensor = np.asarray(tensor)
    factors, final_error, elapsed_time, total_iters, errors, times, best_errors = als_blockwise(tensor=tensor, rank=decomp_rank, random_state=j, method='ELS', tol=tol, n_iter_max=n_iter_max, thresh=thresh, noise_prop=noise_prop, weight=0.5, fitness_threshold=fit_thresh)
    if first_100_plot==True:
        #fits_als[:,j] = 1-torch.tensor(errors)/tl.norm(tensor)
        fits_els[:,j] = [1 - x for x in best_errors/tl.norm(tensor)]
        times_els[:,j] = times
    if noise == True:
        final_error = tl.norm(tensor - tl.kruskal_to_tensor((np.ones(rank), factors)))
        final_errors_els.append(final_error)
    else: 
        final_error = errors[-1]
        final_errors_els.append(final_error)
    final_fits_els.append(1-final_error/tl.norm(tensor))
    final_iters_els.append(total_iters)
    final_times_els.append(elapsed_time)



"""

6. Generate convergence plot. 

"""

if generate_plot:
    # Get the mean fits as at each iter:
    mean_fits_als = np.mean(fits_als,axis=1)
    mean_fits_block_momentum = np.mean(fits_block_momentum,axis=1) 
    mean_fits_block_perturb = np.mean(fits_block_perturb,axis=1)
    mean_fits_block_both = np.mean(fits_block_both,axis=1)
    mean_fits_s1 = np.mean(fits_s1,axis=1) 
    mean_fits_her = np.mean(fits_her,axis=1)
    mean_fits_els = np.mean(fits_els, axis=1)
    mean_fits_ls = np.mean(fits_ls, axis=1)

    # Compute associated standard errors:
    err_als = np.std(fits_als,axis=1)/np.sqrt(nruns)
    err_s1 = np.std(fits_s1,axis=1)/np.sqrt(nruns)
    err_block_both = np.std(fits_block_both,axis=1)/np.sqrt(nruns)
    err_block_momentum = np.std(fits_block_momentum,axis=1)/np.sqrt(nruns)
    err_block_perturb = np.std(fits_block_perturb,axis=1)/np.sqrt(nruns)
    err_her = np.std(fits_her,axis=1)/np.sqrt(nruns)
    err_els = np.std(fits_els,axis=1)/np.sqrt(nruns)
    err_ls = np.std(fits_ls,axis=1)/np.sqrt(nruns)

    # Get the mean times as at each iter:
    mean_times_block_momentum = np.mean(times_block_momentum,axis=1)
    mean_times_block_perturb = np.mean(times_block_perturb,axis=1)
    mean_times_block_both = np.mean(times_block_both,axis=1)
    mean_times_als = np.mean(times_als,axis=1)
    mean_times_s1 = np.mean(times_s1,axis=1)
    mean_times_her = np.mean(times_her,axis=1)
    mean_times_els = np.mean(times_els,axis=1)
    mean_times_ls = np.mean(times_ls,axis=1)

    # Get the cumulative times:
    cum_times_als = np.cumsum(mean_times_als)
    cum_times_s1 = np.cumsum(mean_times_s1)
    cum_times_block_momentum = np.cumsum(mean_times_block_momentum)
    cum_times_block_perturb = np.cumsum(mean_times_block_perturb)
    cum_times_block_both = np.cumsum(mean_times_block_both)
    cum_times_her = np.cumsum(mean_times_her)
    cum_times_els = np.cumsum(mean_times_els)
    cum_times_ls = np.cumsum(mean_times_ls)

    # Convergence plot by time with standard error bars:
    plt.errorbar(cum_times_block_both[1:len(cum_times_block_both)], mean_fits_block_both[1:len(mean_fits_block_both)], yerr=err_block_both[1:len(err_block_both)], c='orange', label='BA-ALS',errorevery=5)
    plt.errorbar(cum_times_block_momentum[1:len(cum_times_block_momentum)], mean_fits_block_momentum[1:len(mean_fits_block_momentum)], yerr=err_block_momentum[1:len(err_block_momentum)], c='red', label='BM-ALS', errorevery=5)
    plt.errorbar(cum_times_block_perturb[1:len(cum_times_block_perturb)], mean_fits_block_perturb[1:len(mean_fits_block_perturb)], yerr=err_block_perturb[1:len(err_block_perturb)], c='green', label='BP-ALS', errorevery=5)
    plt.errorbar(cum_times_s1[1:len(cum_times_s1)], mean_fits_s1[1:len(mean_fits_s1)], yerr=err_s1[1:len(err_s1)], c='purple', label='M-ALS',errorevery=5)
    plt.errorbar(cum_times_als[1:len(cum_times_als)], mean_fits_als[1:len(mean_fits_als)], yerr=err_als[1:len(err_als)], c='blue', label='ALS',errorevery=5)
    plt.errorbar(cum_times_her[1:len(cum_times_her)], mean_fits_her[1:len(mean_fits_her)], yerr=err_her[1:len(err_her)], c='black', label='herALS',errorevery=5)
    plt.errorbar(cum_times_els[1:len(cum_times_els)], mean_fits_els[1:len(mean_fits_els)], yerr=err_els[1:len(err_els)], c='pink', label='ELS',errorevery=5)
    plt.errorbar(cum_times_ls[1:len(cum_times_ls)], mean_fits_ls[1:len(mean_fits_ls)], yerr=err_ls[1:len(err_ls)], c='grey', label='LS',errorevery=5)
    pylab.legend(loc='lower right')
    #plt.ylim([0.9, 0.96])
    #plt.xlim([0, 80])
    plt.xlabel('Time (seconds)')
    plt.ylabel('Fit')
    plt.savefig('convergence.png')
    plt.show()
