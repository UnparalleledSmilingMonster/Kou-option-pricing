import numpy as np
from scipy.special import factorial, binom, erf, ndtr  # ndtr is the CDF for the normal distribution
from scipy.integrate import quad
from scipy.optimize import root_scalar

import mpmath as mp
#np.seterr(all='warn')

#Math variables
pi = np.pi
precision_type = np.longdouble
mp.mp.dps = 30


"""
TODO : modify accordingly
We evaluate probability of asset X_t = log(S_t/S_0) > a at maturity T, under the parameters precised above. 
2 methods : 1) Kou's analytic resolution
            2) Monte Carlo simulation  
"""



#Computing Hermite integral
def Hhn(x, n):
    result, _ = quad(lambda t : (t-x)**n * np.exp(-t**2 / 2), x, np.inf)
    return result / factorial(n)
    
        
#Computing Hermite integral using recurrence relation for faster computation:
def Hhn_rec(x,m):
    hermite_minus1 = lambda t : np.exp(-t**2/2)
    hermite_0 = lambda t : np.sqrt(2*pi) 
    
    hermite = np.zeros(m+1, dtype=precision_type)
    hermite[0] = hermite_0(x)
    hermite[1] = hermite_minus1(x) - x*hermite[0]
    for n in range(2,m+1):
        hermite[n] = 1/n * hermite[n-2] - x/n * hermite[n-1]    
    return hermite
      
    
def binomial(m):
    #Store 2d array of binomial coefficients
    global binomial_arr
    binomial_arr = np.zeros((m+1,m+1), dtype=float) #as binomial[k][n]
    
    for n in range(m+1):
        for k in range(n+1):
            binomial_arr[k,n] = binom(n,k)
            
def eta_powers(eta_1, eta_2, m):
    global arr1, arr2           
    #Store array of powers of eta_1 / (eta_1 + eta_2) 
    u1 = eta_1 / (eta_1 + eta_2)
    arr1 = np.array([u1**k for k in range(m+1)])

    #Store array of powers of eta_2 / (eta_1 + eta_2) 
    u2 = eta_2 / (eta_1 + eta_2)
    arr2 = np.array([u2**k for k in range(m+1)])    
    return arr1, arr2    

def q_p_powers(p, q, m):
    q = 1-p
    global p_pow, q_pow
    p_pow = np.array([p**k for k in range(m+1)])
    q_pow = np.array([q**k for k in range(m+1)])
    
def P_and_Q(eta_1, eta_2, p, m):
    q = 1-p
    #Computation of P(n,k) and Q(n,k):
    P_arr = np.zeros((m+1, m+1), dtype=precision_type)
    Q_arr = np.zeros((m+1, m+1), dtype=precision_type)
    for n in range(m+1):
        for k in range(1,n+1):
            if k==n :
                P_arr[n,k] = p**n
                Q_arr[n,k] = q**n
            else:
                i_range = np.arange(k,n) #from k to n-1
                P_arr[n,k] = np.sum(binomial_arr[i_range-k, n-k-1] * binomial_arr[i_range, n] * arr1[i_range-k]*arr2[n-i_range] * p_pow[i_range] * q_pow[n-i_range])
                Q_arr[n,k] = np.sum(binomial_arr[i_range-k, n-k-1] * binomial_arr[i_range, n] * arr1[n-i_range]*arr2[i_range-k] * p_pow[n-i_range] * q_pow[i_range])
    
    return P_arr, Q_arr


#Computing the I_k for B>0, A =! 0
def In_pos(c, a, b, d, m):
    arr = np.zeros(m, dtype=precision_type)
    hh = np.array([Hhn(b*c -d, k) for k in range(m+1)])
    #computing repetitive terms: their idx is arranged accordingly to their precedence in the formula
    t1 = np.exp(a*c)/a
    t2 = np.sqrt(2*pi)/b * np.exp(a*d/b + a**2/(2*b**2))

    b_over_a = np.array([(b/a)**k for k in range(m+1)])
    for n in range(m):
        i_range = np.arange(0, n+1)
        arr[n] = -t1 * np.sum(b_over_a[n-i_range] * hh[i_range]) + b_over_a[n+1]*t2*ndtr(-b*c+d+a/b) 
    return arr    


#Computing the I_k for B<0, A < 0
def In_neg(c, a, b, d, m):
    arr = np.zeros(m, dtype=precision_type)
    hh = np.array([Hhn(b*c -d, k) for k in range(m+1)])
    #computing repetitive terms: their idx is arranged accordingly to their precedence in the formula
    t1 = np.exp(a*c)/a
    t2 = np.sqrt(2*pi)/b * np.exp(a*d/b + a**2/(2*b**2))

    b_over_a = np.array([(b/a)**k for k in range(m+1)])
    for n in range(m):
        i_range = np.arange(0, n+1)
        arr[n] = -t1 * np.sum(b_over_a[n-i_range] * hh[i_range]) - b_over_a[n+1]*t2*ndtr(b*c - d - a/b) 
    return arr    


def chi(mu, sigma, lambda_, p, eta_1, eta_2, a, T, m, debug = False):
    #Reccurent terms:
    sqrt_T = np.sqrt(T)
    q = 1-p     
        
    q_p_powers(p, q, m)
    arr1, arr2 = eta_powers(eta_1, eta_2, m)
    
    #P_nk and Q_nk:
    P_arr, Q_arr = P_and_Q(eta_1, eta_2, p, m)

    #Probability of Poisson = k : sufficiently low nb of terms not to compute it 'efficiently'
    poiss = np.array([np.exp(-lambda_ *T) * (lambda_ *T)**k / factorial(k) for k in range(m+1)])  
    
    #I_k (2 arrays : 1 for pos param and 1 for neg params)
    In_pos_arr = In_pos(a - mu*T, eta_2, 1/(sigma * sqrt_T), - sigma * eta_2 * sqrt_T, m)  
    In_neg_arr = In_neg(a - mu*T, -eta_1, -1/(sigma * sqrt_T), - sigma * eta_1 * sqrt_T, m) 
  
    if debug:
        print(Hhn(-4,5))     
        print("P52:", P_arr[5,2])
        print("Q52:", Q_arr[5,2])
        print("I_5(-2,-2,-2,-2):",In_neg(-2,-2,-2,-2,10)[5])   
        print("I_5(2,2,2,2):",In_pos(2,2,2,2,10)[5])    
        
    
    #Chi computation: 
    sigma_eta1 = np.array([(sigma*eta_1*sqrt_T)**k for k in range(m+1)])
    sigma_eta2 = np.array([(sigma*eta_2*sqrt_T)**k for k in range(m+1)])

    temp_P = 0
    temp_Q = 0    
    for n in range(1,m+1):
        temp_P += poiss[n] * np.sum(P_arr[n, 1:n+1] * sigma_eta1[1:n+1] * In_neg_arr[0:n])
        temp_Q += poiss[n] * np.sum(Q_arr[n, 1:n+1] * sigma_eta2[1:n+1] * In_pos_arr[0:n])
    
    chi = np.exp((sigma*eta_1)**2 * T /2) / (sigma * np.sqrt(2*pi*T)) * temp_P \
        + np.exp((sigma*eta_2)**2 * T /2) / (sigma * np.sqrt(2*pi*T)) * temp_Q \
        + poiss[0] * ndtr(- (a-mu *T)/(sigma * sqrt_T))
        
    
    """
    #Code version for debug (nested for loops less efficient)
    chi = 0
    for n in range(1, m+1):
        temp_P = 0
        temp_Q = 0
        for k in range(1, n+1):
            temp_P += P_arr[n,k] * sigma_eta1[k] * In_neg_arr[k-1]
            temp_Q += Q_arr[n,k] * sigma_eta2[k] * In_pos_arr[k-1]
         
        chi +=  np.exp((sigma*eta_1)**2 * T /2) / (sigma * np.sqrt(2*pi*T))  * temp_P * poiss[n] +  np.exp((sigma*eta_2)**2 * T /2) / (sigma * np.sqrt(2*pi*T))  *temp_Q*poiss[n]
    chi+=   poiss[0] * ndtr(- (a-mu *T)/(sigma * sqrt_T))  
    """
    return chi

#########################################################################################

def european_call_option(S0, r, sigma, lambda_, p, eta_1, eta_2, K, T, m):
    q = 1-p
    zeta = p*eta_1/(eta_1 - 1) + q*eta_2/(eta_2 +1) - 1
    mu =  r - sigma**2/2 - lambda_ *zeta    
    a = np.log(K/S0)

    p_tilde = p/(1+zeta) *eta_1/(eta_1 - 1)
    mu_tilde= r +  sigma**2/2 - lambda_*zeta
    eta_1_tilde = eta_1 -1
    eta_2_tilde = eta_2 +1
    lambda_tilde = lambda_*(zeta + 1)
    
    return S0 * chi(mu_tilde, sigma, lambda_tilde, p_tilde, eta_1_tilde, eta_2_tilde, a, T, m)  \
        -  K * np.exp(-r*T) * chi(mu, sigma, lambda_, p, eta_1, eta_2, a, T, m)
        


if __name__ == '__main__':

    # Parameters in Kou's paper from 2002 under European's options pricing section
    eta_1 = 10
    eta_2 = 5
    lambda_ = 1.0
    p = 0.4
    #q = 0.6
    r = 0.05
    sigma = 0.16
    K = 98
    T = 0.5
    S0 = 100
    

    m = 20 #number of points before truncation of the infinite sums 
    binomial(m) #we can compute the binomial coefficients as a global variable beforehand
    
    print("Price of the european call option:", european_call_option(S0, r, sigma, lambda_, p, eta_1, eta_2, K, T, m))    
    
    
    
