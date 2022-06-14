import numpy as np
from numba import njit, vectorize

@njit
def sample_proposal(mu, sigma):
    """
    This function returns as an output a random number sampled from a distribution.
    This distribution has to depend on the current state of the Markov chain, at the moment the state is the mean of the
    Gaussian 'mu'. 'sigma' is a fixed parameter.
    
    """
    return np.random.normal(mu, sigma)

@vectorize
def posterior_dist(x):
    """
    This function computes the value of the unnormalized posterior distribution from which we want to
    sample the random numbers. In the context of GW astronomy it is the product of the Likelihood and the Priors.
    
    """
   
    if 0 <= x <= 1:
        return x**2 * ( 1 - x**2 )**(7/2)
    else:
        return 0
    
    
@njit
def run_sampler(starting_point, n_max, burn_in, sigma, seed = None):
    """
    This function runs the Metropolis-Hastings algorithm to sample from the distribution contained in posterior_dist().
    
    Input
        starting_point:         float, starting point of the Markov chain
        n_max         :         float, maximum number of cycles of the MH algorithm
        burn_in       :         int, number of entries of the Markov chain to discard
        sigma         :         float, sqrt of the variance of the gaussian, used in sample_proposal()

    Output 
        samples       :         list, list of samples from the posterior distribution
        numb_accept   :         float, number of accepted unique samples
    """
    
    if seed != None:
        np.random.seed(seed)
    
    samples = np.zeros(n_max)
    numb_accept = 0
    
    samples[0] = starting_point
    for idx in range(n_max):
        
        proposal = sample_proposal(samples[idx], sigma)
        
        accept_prob = min(1,
                          posterior_dist(proposal) / posterior_dist(samples[idx])
                         )
    
        if np.random.uniform(0,1) < accept_prob:
            samples[idx+1] = proposal
        else:
            samples[idx+1] = samples[idx]

    return samples[burn_in + 1:]