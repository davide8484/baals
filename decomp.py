# Import required packages:
import numpy as np
import warnings
import tensorly as tl
import torch 
import time
import scipy as sp


def check_random_state(seed):
    """Returns a valid RandomState
    Parameters
    ----------
    seed : None or instance of int or np.random.RandomState(), default is None
    Returns
    -------
    Valid instance np.random.RandomState
    Notes
    -----
    Inspired by the scikit-learn eponymous function
    """
    if seed is None or isinstance(seed, int):
        return np.random.RandomState(seed)

    elif isinstance(seed, np.random.RandomState):
        return seed

    raise ValueError('Seed should be None, int or np.random.RandomState')


def initialize_factors(tensor, rank, init, svd, random_state, non_negative):
    r"""Initialize factors used in `parafac`.
    The type of initialization is set using `init`. If `init == 'random'` then
    initialize factor matrices using `random_state`. If `init == 'svd'` then
    initialize the `m`th factor matrix using the `rank` left singular vectors
    of the `m`th unfolding of the input tensor.
    Parameters
    ----------
    tensor : ndarray
    rank : int
    init : {'svd', 'random'}, optional
    svd : str, default is 'numpy_svd'
        function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS
    non_negative : bool, default is False
        if True, non-negative factors are returned
    Returns
    -------
    factors : ndarray list
        List of initialized factors of the CP decomposition where element `i`
        is of shape (tensor.shape[i], rank)
    """
    rng = check_random_state(random_state) 

    if init == 'random':
        factors = [tl.tensor(rng.random_sample((tensor.shape[i], rank)), **tl.context(tensor)) for i in range(tl.ndim(tensor))]
        if non_negative:
            return [tl.abs(f) for f in factors]
        else:
            return factors

    elif init == 'svd':
        try:
            svd_fun = tl.SVD_FUNS[svd]
        except KeyError:
            message = 'Got svd={}. However, for the current backend ({}), the possible choices are {}'.format(
                    svd, tl.get_backend(), tl.SVD_FUNS)
            raise ValueError(message)

        factors = []
        for mode in range(tl.ndim(tensor)):
            U, _, _ = svd_fun(tl.unfold(tensor, mode), n_eigenvecs=rank)

            if tensor.shape[mode] < rank:
                random_part = tl.tensor(rng.random_sample((U.shape[0], rank - tl.shape(tensor)[mode])), **tl.context(tensor))
                U = tl.concatenate([U, random_part], axis=1)
            if non_negative:
                factors.append(tl.abs(U[:, :rank]))
            else:
                factors.append(U[:, :rank])
        return factors

    raise ValueError('Initialization method "{}" not recognized'.format(init))


def ALS(tensor, factors, rank):
    
    """ 
    
    This function performs one full ALS iteration (updates all factor matrices).
    
    Inputs
    -------
    - factors: list containing the factor matrices. Each factor matrix is a numpy array
    - rank: the chosen rank for the tensor decomposition
    
    Outputs
    -------
    - curr_factors: the new set of factor matrices after applying an ALS step 
    to 'factors'
    
    
    """
    
    for mode in range(tl.ndim(tensor)): 
        pseudo_inverse = tl.tensor(np.ones((rank, rank)), **tl.context(tensor))
        for i, factor in enumerate(factors):
            if i != mode:
                pseudo_inverse = pseudo_inverse*tl.dot(tl.transpose(factor), factor)
        factor = tl.dot(tl.unfold(tensor, mode), tl.tenalg.khatri_rao(factors, skip_matrix=mode))
        factor = tl.transpose(tl.solve(tl.transpose(pseudo_inverse), tl.transpose(factor)))
        factors[mode] = factor 
    curr_factors = factors.copy() # Updated factor matrices
    return curr_factors


def remove_factor_collinearity(U, collinearities, rank, thresh, noise_prop, noise_injections):

    """ 
    
    This function takes a factor matrix and, if any two factors (matrix columns)
    have collinearity above a specified threshold ('thresh'), reduces the 
    pair's collinearity by replacing elements of one factor with random noise.
    
    Inputs 
    -------
    - U: the factor matrix
    - collinearities: the collinearity between each pair of factors in the matrix
    - rank: the rank of the tensor decomposition
    - thresh: the collinearity threshold at which to inject the noise. Typical 
    values are 0.9-0.99
    - noise_prop: the proportion of factor elements to replace with random noise
    - noise_injections: tracks the number of noise injections
    
    Outputs
    -------
    - U: the factor matrix after noise has been injected (if noise is required)
    - noise_injections: counts the number of noise injections
    
    """
    
    # Number of elements in each factor to replace with noise:
    num_elem = int(np.max([1,np.round(noise_prop*np.size(U[:,1]))]))

    for g in range(0,rank):
        if abs(np.max(collinearities[g,:]))>thresh: # If the gth factor has collinearity above threshold with at least one other factor...
            #U[0:num_elem,g] = np.random.uniform(U[:,g].min(), U[:,g].max(), len(U[0:num_elem,g])) # Replace part of the gth factor with noise
            U[:,g] = perturb(U[:,g], num_elem)
            noise_injections = noise_injections+1
    
    return U, noise_injections


def perturb(a, m):
    
    """
    
    This function replaces m randomly selected elements from the vector a with 
    random noise. The vector a is a factor from the factor matrix.
    
    """
    
    idx = np.random.choice(len(a), m)
    v0, v1 = a.min(), a.max()
    a[idx] = v0 + (v1-v0)*np.random.rand(m)
    return a


def minR(R, Aprev, Bprev, Cprev, Ga, Gb, Gc, tensor):
    
    """
    
    This function implements the enhanced line search method to find the 
    optimal R in each ALS iteraction. The optimal R is found via a call to 
    the scipy minimize function. 
    
        
    Inputs
    -------
    - R: the weight for enhanced line search.
    - Aprev, Bprev, Cprev: factor matrices from prior iteration. 
    - Ga, Gb, Gc: current factor matrix less prior factor matrix.
    - tensor: the true tensor.

    Outputs
    -------
    - total_err: the approximation error for a given value of R. 
    - Note: the function is used in a call to the scipy minimize function, 
    which searches for the value of R that minimizes the approximation 
    error. 

    
    """
    
    # Get the bracketed elements of error function:
    el1 = Aprev + R * Ga
    el2 = Bprev + R * Gb
    el3 = Cprev + R * Gc
    
    xijk = tl.unfold(tensor, mode=0) # Unfold tensor along first mode
    hj = np.matmul(el1, np.transpose(sp.linalg.khatri_rao(el2, el3)))
    
    total_err = np.linalg.norm(xijk - hj, ord='fro') # Compute Frob norm
    
    return total_err





def als_blockwise(tensor, rank, random_state, method, tol, n_iter_max, thresh, noise_prop, weight, fitness_threshold):
    
    """
    
    This function decomposes a rank N tensor using the specified method. The 
    following methods are available: 
        - 'ALS' (standard ALS)
        - 'LS' (line search ALS)
        - 'ELS' (enahnced line search ALS)
        - 'ALS_momentum' (see Mitchell et al 2018 - 'Nesterov acceleration of ALS
        for CP decomposition')
        - 'momentum': ALS with blockwise momentum
        - 'perturb': ALS with blockwise perturbation
        - 'momentum_and_perturb': blockwise accelerated ALS. I.e. ALS with 
        blockwise momentum and perturbation
        - 'herALS'
        
    Inputs
    -------
    - tensor: the tensor to be decomposed
    - rank: the rank of the decomposition
    - random_state: random start point for the decomposition
    - method: the decomposition method to apply - see the methods listed above
    - tol: reconstruction error tolerance for termination criterion
    - n_iter_max: the maximum number of iterations to run the algorithm
    - thresh: the collinearity threshold for injecting noise. Typical values 
    are 0.9-0.99
    - noise_prop: the proportion of factor elements to replace with random 
    noise
    - weight: the momentum weight to use for the blockwise methods.
    - fitness_threshold: the fitness score required for the algorithm to have 
    'converged'. In our experiments, we set this to be within 0.01 of the best 
    expected fit (e.g. 0.99 in the case of noiseless simulated tensors). 
    
    Outputs
    -------
    - factors: the factor matrices obtained via the decomposition
    - final_error: the reconstruction error upon termination
    - elapsed_time: the total run time of the decomposition
    - total iters: the total number of iterations for the decomposition
    - times: the time taken for each iteration in the decomposition

    
    """
    
    tl.set_backend('numpy')
    
    times = [] # Store run times
    a = time.time()
    
    # Randomly generate initial factor matrices:
    factors = initialize_factors(tensor, rank, init='random', svd='numpy_svd', random_state=random_state, non_negative=False)
    
    # Initialise parameters and storage:    
    noise_injections = 0
    beta = 0
    betas = []
    betas.append(beta)
    errors = []
    best_errors = []
    
    errors.append(tl.norm(tensor - tl.kruskal_to_tensor((np.ones(rank), factors))))
    best_errors.append(tl.norm(tensor - tl.kruskal_to_tensor((np.ones(rank), factors))))
    times.append(time.time() - a)
    
    # Initialise parameters for herALS:
    beta_zero = 0.5
    eta = 1.5
    gamma_hat = 1.01
    gamma = 1.05
    
    a = time.time()
    
    # Line search parameters:
    acc_pow = 2.0  # Extrapolate to the iteration^(1/acc_pow) ahead
    acc_fail = 0  # How many times acceleration have failed
    max_fail = 4  # Increase acc_pow with one after max_fail failure
    
    
    # Do first ALS iteration:
    factors = ALS(tensor=tensor, factors=factors, rank=rank)
    errors.append(tl.norm(tensor - tl.kruskal_to_tensor((np.ones(rank), factors))))
    best_errors.append(tl.norm(tensor - tl.kruskal_to_tensor((np.ones(rank), factors))))
    tensor_norm = tl.norm(tensor) # Tensor norm
    num_modes = tl.ndim(tensor) # Number of modes

    # Initialise the acc_term object:
    acc_term = factors

    # Set the schedule for injecting the noise:
    schedule = [3,4,6,10,18,34] # Corresponds to 2**t + 1 for t=1,...,6
    noise_injections = 0 # Count the number of noise injections

    # Counter for number of restarts (no longer used):
    restarts = 0

    # Set previous iteration's factors:
    prev_factors = factors[:]

    times.append(time.time() - a)
    start_time = time.time()

    for k in range(3,n_iter_max):
        
        if method == 'ALS':
            a = time.time()
            factors = ALS(tensor=tensor, factors=factors, rank=rank)
            error = tl.norm(tensor - tl.kruskal_to_tensor((np.ones(rank), factors)))
            errors.append(error)
            times.append(time.time() - a)
        
        elif method == 'LS':
            # For the first few iterations, do normal ALS updates:
            if k<6:
                a = time.time()
                factors = ALS(tensor=tensor, factors=factors, rank=rank)
                error = tl.norm(tensor - tl.kruskal_to_tensor((np.ones(rank), factors)))
                errors.append(error)
                times.append(time.time() - a)
            # Then do LS updates:
            else: 
                a = time.time()
                
                # First, copy the prev factors:
                factors_last = [tl.copy(f) for f in factors]
                
                # Second, do ALS update step to get new factors:
                factors = ALS(tensor=tensor, factors=factors, rank=rank)
                
                
                # Third, do LS step:
                jump = k ** (1.0 / acc_pow)

                # Generate new factors under LS:
                new_factors = [
                    factors_last[ii] + (factors[ii] - factors_last[ii]) * jump
                    for ii in range(tl.ndim(tensor))
                ]
                
                # Compute reconstruction error of the proposed LS solution:
                new_rec_error = tl.norm(tensor - tl.kruskal_to_tensor((np.ones(rank), new_factors)))
                
                # If this reconstruction error is less than the previous error, update factors and store new error:
                if new_rec_error < errors[-1]:
                    factors = new_factors
                    acc_fail = 0
                    errors.append(new_rec_error)
                
                # If reconstruction error is not less than previous error, do not update factors with LS solution, compute error of current solution, and store:
                else:
                    error = tl.norm(tensor - tl.kruskal_to_tensor((np.ones(rank), factors)))
                    errors.append(error)
                    acc_fail += 1
                    if acc_fail == max_fail:
                        acc_pow += 1.0
                        acc_fail = 0
                
                times.append(time.time() - a)
            
        elif method == 'ELS':
            a = time.time()
            # Generate new set of factors:
            if k<2:
                factors = ALS(tensor=tensor, factors=factors, rank=rank)
            else:
                
                # STEP 1 IN THE PAPER:
                
                # Factor matrices from the previous iteration:
                Aprev = prev_factors[0]
                Bprev = prev_factors[1]
                Cprev = prev_factors[2]
                
                # Obtain the G matrices (differences between the current and previous factors):
                Ga = factors[0] - Aprev
                Gb = factors[1] - Bprev
                Gc = factors[2] - Cprev
                
                # ELS step: search for the value of R that minimizes the approximation error:
                els_search = sp.optimize.minimize(fun = minR, x0 = 0, args = (Aprev, Bprev, Cprev, Ga, Gb, Gc, tensor))    
                ropt = els_search.x # Optimal value of R
                
                # STEP 2 IN THE PAPER:
                
                # Compute Anew, Bnew, Cnew:
                Anew = Aprev + ropt * Ga
                Bnew = Bprev + ropt * Gb
                Cnew = Cprev + ropt * Gc
                
                # STEP 3 IN THE PAPER:
                
                # Apply the ALS step using the new factors:
                factors_new = [Anew,Bnew,Cnew]
                factors = ALS(tensor=tensor, factors=factors_new, rank=rank)
              
                # Set Anew, Bnew, Cnew as the previous iteration values:
                prev_factors = [Anew,Bnew,Cnew]

                # STEP 4 IN THE PAPER:
                    
                # Calculate approximation error:
                #error = tl.norm(tensor - tl.kruskal_to_tensor((np.ones(rank), factors)))
            
            # Store the errors and times:
            error = tl.norm(tensor - tl.kruskal_to_tensor((np.ones(rank), factors)))
            errors.append(error)
            times.append(time.time() - a)
            
            
        elif method == 'ALS_momentum':
            a = time.time()
            
            # Do normal ALS for early iterations:
            if k<5:
                factors = ALS(tensor=tensor, factors=factors, rank=rank)
            else:
                # Restart condition:
                if errors[-1]>errors[-2] and beta !=0:
                    factors = prev_factors
                    beta = 0
                else:
                    beta = 1
                # Generate momentum term:
                for i in range(num_modes):
                    acc_term[i] = factors[i] + beta*(factors[i] - prev_factors[i])
                    
                # Update previous factors for next iter:
                prev_factors = factors
                
                # Do the ALS update:
                factors = ALS(tensor=tensor, factors=acc_term, rank=rank)
                
            error = tl.norm(tensor - tl.kruskal_to_tensor((np.ones(rank), factors)))
            errors.append(error)    
            times.append(time.time() - a)   
        
        elif method == 'herALS':
            a = time.time()
            
            # Initial iteration:
            if k==3:
                beta = beta_zero
                beta_hat = beta_zero # Max bete
                beta_prev = beta
                
            else: 
        
                # Restart condition - reset factor matrices to previous iteration:
                if errors[-1]>errors[-2]:
                    factors = prev_factors # Set factors as previous factors
                    beta_hat = beta_prev  # Set the max beta as the previous beta
                    beta = beta / eta # Decrease beta
                    
                else: 
                    beta_hat = min(1, beta_hat * gamma_hat)
                    beta = min(beta_hat, beta_prev * gamma)
            
            beta_prev = beta # Store the previous iteration's beta
            
            # Apply the blockwise method:
            for mode in range(tl.ndim(tensor)): 
                
                # Extract the mode's factor matrix in the previous iteration (for use in the momentum term):
                prev_factor = factors[mode][:]
                
                # Do an ALS step to update the current mode's factor matrix:
                pseudo_inverse = tl.tensor(np.ones((rank, rank)), **tl.context(tensor))
                for i, factor in enumerate(factors):
                    if i != mode:
                        pseudo_inverse = pseudo_inverse*tl.dot(tl.transpose(factor), factor)
                factor = tl.dot(tl.unfold(tensor, mode), tl.tenalg.khatri_rao(factors, skip_matrix=mode))
                factor = tl.transpose(tl.solve(tl.transpose(pseudo_inverse), tl.transpose(factor)))
                
                # Store the factor matrix obtained from the ALS step:
                als_factor = factor[:]
                
                
                # Apply herALS:
                factor = factor + beta*(factor - prev_factor)
                # Store the factors:
                factors[mode] = factor 
                    
                factors_evaluate = factors[:]
                    
                # Do proper evaluation (do not add momentum term for the last sub-step in ALS update fo
                # for purpose of evaluating error; do add for further udpates)
                if mode == tl.ndim(tensor):
                    factors_evaluate[mode] = als_factor
                    
            error = tl.norm(tensor - tl.kruskal_to_tensor((np.ones(rank), factors_evaluate)))
            errors.append(error)
    
            times.append(time.time() - a)       
            
        
        else:
            
            # Blockwise methods are applied below (mode by mode).
            
            a = time.time()
            
            for mode in range(tl.ndim(tensor)): 
                
                # Extract the mode's factor matrix in the previous iteration (for use in the momentum term):
                prev_factor = factors[mode][:]
                
                # Do an ALS step to update the current mode's factor matrix:
                pseudo_inverse = tl.tensor(np.ones((rank, rank)), **tl.context(tensor))
                for i, factor in enumerate(factors):
                    if i != mode:
                        pseudo_inverse = pseudo_inverse*tl.dot(tl.transpose(factor), factor)
                factor = tl.dot(tl.unfold(tensor, mode), tl.tenalg.khatri_rao(factors, skip_matrix=mode))
                factor = tl.transpose(tl.solve(tl.transpose(pseudo_inverse), tl.transpose(factor)))
                
                # Store the factor matrix obtained from the ALS step:
                als_factor = factor[:]
                
                # Below: apply the momentum and/or perturbation step to the above ALS factor:
                
                # (A) Perturbation method:
                
                if method == 'perturb' and k in schedule[1:len(schedule)]:
                    # Get the correlation matrix for the factors in the ALS factor:
                    corr_matrix = np.corrcoef(np.transpose(als_factor))
                    # Get upper triangle matrix containing the correlations:
                    collinearities = np.triu(corr_matrix,k=1)
                    # Perturb highly collinear columns:
                    noise_prop = 1/(k-2)
                    perturbed_factor, noise_injections = remove_factor_collinearity(U=als_factor, collinearities = collinearities, rank=rank, thresh=thresh, noise_prop=noise_prop, noise_injections=noise_injections)
                    # Update factor objects:
                    factor = perturbed_factor
                    factors[mode] = factor
                    factors_evaluate = factors[:]
                    
                    # Create factors object for computing the error after one full 
                    # step (all matrices updated). To do this, we do not add the 
                    # momentum/perturbation term to the last (Nth) mode:
                    if mode == tl.ndim(tensor):
                        factors_evaluate[mode] = als_factor
                    
                elif method == 'perturb' and k not in schedule[1:len(schedule)]:
                    factor = als_factor
                    factors[mode] = factor
                    factors_evaluate = factors[:]
                    if mode == tl.ndim(tensor):
                        factors_evaluate[mode] = als_factor
                
                # (B) Momentum method:
                
                elif method == 'momentum':
                    # Generate weight:
                    if k<5:
                        beta=0
                    else:
                        beta = weight
        
                    # Add momentum to factor:
                    factor = factor + beta*(factor - prev_factor)
                    # Store the factors:
                    factors[mode] = factor     
                    factors_evaluate = factors[:]
                    
                    # Create factors object for computing the error after one full 
                    # step (all matrices updated). To do this, we do not add the 
                    # momentum/perturbation term to the last (Nth) mode:
                    if mode == tl.ndim(tensor):
                        factors_evaluate[mode] = als_factor
                   
                # (C) Momentum and perturbation, i.e. BA-ALS:
                    
                elif method == 'momentum_and_perturb' and k in schedule[1:len(schedule)]:
                    
                    if k<5:
                        beta=0
                        
                    else:
                        beta = weight

                    # First add momentum term:
                    factor = factor + beta*(factor - prev_factor)
                    # Then do the same perturbation process as above under the perturbation method:
                    corr_matrix = np.corrcoef(np.transpose(factor))
                    collinearities = np.triu(corr_matrix,k=1)
                    noise_prop = 1/(k-2)
                    perturbed_factor, noise_injections = remove_factor_collinearity(U=factor, collinearities = collinearities, rank=rank, thresh=thresh, noise_prop=noise_prop, noise_injections = noise_injections)
                    factor = perturbed_factor
                    factors[mode] = factor
                    factors_evaluate = factors[:]
                    
                    # Create factors object for computing the error after one full 
                    # step (all matrices updated). To do this, we do not add the 
                    # momentum/perturbation term to the last (Nth) mode:
                    if mode == tl.ndim(tensor):
                        factors_evaluate[mode] = als_factor
                    
                elif method == 'momentum_and_perturb' and k not in schedule[1:len(schedule)]: 
                    
                    if k<5:
                        beta=0
                    else:
                        beta = weight

                    factor = factor + beta*(factor - prev_factor)
                    # Store the factors:
                    factors[mode] = factor 
                    
                    factors_evaluate = factors[:]
                    
                    # Do proper evaluation (do not add momentum term for the last sub-step in ALS update fo
                    # for purpose of evaluating error; do add for further udpates)
                    if mode == tl.ndim(tensor):
                        factors_evaluate[mode] = als_factor
                
                
                
            error = tl.norm(tensor - tl.kruskal_to_tensor((np.ones(rank), factors_evaluate)))
            
            if k in schedule and method in ['momentum', 'momentum_and_perturb', 'perturb']:
                errors.append(errors[-1])
                
            elif k in [x+1 for x in schedule] and method in ['momentum', 'momentum_and_perturb', 'perturb']:
                errors.append(error)
                errors[-2] = np.mean([errors[-1],errors[-2]]) # Smooth

            else:
                errors.append(error)
    
            times.append(time.time() - a)
    
        best_errors.append(np.min(errors))
    
        # Stopping criterion:
        if abs(errors[-1] - errors[-2])/tensor_norm < tol:
            break
        
        fitt = 1- best_errors[-1]/tensor_norm
        
        # If fitness threshold is met, the alg has converged:    
        if fitt > fitness_threshold:
            break
        
        betas.append(beta)
        
    final_error = best_errors[-1]
    elapsed_time = time.time() - start_time
    total_iters = k + restarts
        
    return factors, final_error, elapsed_time, total_iters, errors, times, best_errors



def simulate_tensors2(random_seed, I, a, b, R, l1, l2, bottleneck, torch_tensor):
    
    """
    This function is based on the method used in Tomasi and Bro (2006) to simulate 
    cubic tensors with varying degrees of ill-conditioning. It can produce both
    'bottleneck' scenarios (where two or more factors in one mode are collinear) 
    and 'swamp' scenarios (where there is a bottleneck in each mode). Note that 
    the bottlenecks and swamps that this function generates involve all factors 
    in a given mode being collinear. 
    Inputs:
        1. random_seed: used for reproducibility of the tensors.
        2. I: the size of each tensor is IxIxI.
        3. a and b: the collinearity of the factors in mode n is drawm uniformly
        at random on (a[n], b[n]). So set [a[1],a[2],a[3]] and [b[1],b[2],b[3]]
        such that 0 <= a[n] <= b[n] <= 1. 
        4. R: the true rank of the tensor.
        5. l1: the level of homoscedastic noise.
        6. l2: the level of heteroscedastic noise.
        7. bottleneck: set at 0 for a swamp (collinear factors in all 3 modes); 
        set at 1 or 2 for a bottleneck (collinear factors in 1 or 2 modes 
        respectively).
        8. torch_tensor: set as False to generate a numpy array (for input into 
        decomp function); set to True to generate a torch tensor (for input into 
        decomp_ls function)
    Output:
        1. Z: the simulated tensor with true rank R, size IxIxI, and some specified 
        level of collinearity between the factors of each mode.
        2. Zprime: Z, potentially with some level l1 of homoscedastic noise added.
        3. Zdprime: Zprime, potentially with some level l2 of heteroscedastic 
        noise added
        4. U: the true factor matrices.
    """
    
    
    tl.set_backend('numpy')
    np.random.seed(random_seed) 
    
    U = np.empty([3, I, R]) # Create empty array to store factor matrices
        
    for n in range(0,3):
        collinearity = np.random.uniform(a[n],b[n])
        
        # First generate K:
        K = collinearity*np.ones((R,R))+(1-collinearity)*np.identity(R)
        
        # Get C as Cholesky factor of K (transpose returns upper triangle matrix):
        C = np.transpose(np.linalg.cholesky(K))
        
    # Generate the factor matrices:
    #for n in range(0,3):
        M = np.random.normal(0,1,[I,R]) # Generate a random matrix
        Q = np.linalg.qr(M)[0] # Ortho-normalize the columns of M to give Q
        U[n] = np.matmul(Q,C) # Store the factor matrices
    
    # Adjust the factor matrices to generate a bottleneck if specified:
    if bottleneck==1:
        U[1] = np.random.normal(0,np.std(U),[I,R])
        U[2] = np.random.normal(0,np.std(U),[I,R])
    elif bottleneck==2:
        U[2] = np.random.normal(0,np.std(U),[I,R])
    
    # Turn the Khatri-product of factor matrices into full tensor:
    Z = tl.kruskal_to_tensor((np.ones(R),U))
    
    
    # Generate two random normal tensors for the purpose of adding noise to Z:
    N1 = np.random.normal(0,1,[I,I,I])
    N2 = np.random.normal(0,1,[I,I,I])
    nZ = np.linalg.norm(Z) # Norm of Z
    nN1 = np.linalg.norm(N1) # Norm of N1
    
    # Generate Zprime by adding homoscedastic noise to Z:
    if l1>0:
        Zprime = Z + 1/np.sqrt(100/l1-1)*nZ/nN1*N1
        nZprime = np.linalg.norm(Zprime)
    else: 
        Zprime = Z
        nZprime = nZ
    
    N2Zprime = np.multiply(N2,Zprime)
    nN2Zprime = np.linalg.norm(N2Zprime)
    
    # Generate Zdprime by adding heteroscedastic noise to Z:
    if l2>0:
        Zdprime = Zprime + 1/np.sqrt(100/l2 - 1)*nZprime/nN2Zprime*N2Zprime
    else: 
        Zdprime = Zprime
        
    if torch_tensor==True:
        Z = torch.from_numpy(Z)
        Zprime = torch.from_numpy(Zprime)
        Zdprime = torch.from_numpy(Zdprime)
        
    return Z, Zprime, Zdprime, U
