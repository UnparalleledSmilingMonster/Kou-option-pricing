import numpy as np
from scipy.special import factorial, binom, erf, ndtr  # ndtr is the CDF for the normal distribution


from tqdm import tqdm
import mpmath as mp



#Math variables
pi = mp.pi
mp.mp.dps = 70


"""
The Hi functions are computed from the Hh functions. The recurrence relation to compute the Hh functions can be leveraged into finding a recurrence relation to compute the Hi functions. As such, we would only need to compute H-1(a,b,c,n) and H0(a,b,c,n).
However, there is also a way to compute H0 from H-1. So we can express all Hi functions as H-1 functions, hence only ever computing H-1.

Hh_n(x) = 1/n! * int_x^\infty (t-x)^n *exp(-t^2/2) dt
H_i(a,b,c,n) = 1/sqrt(2.pi) * int_0^\infty exp[(0.5*c^2 -b)*t] * t^(n+i/2) Hh_i(c*sqrt(t) + a /sqrt(t)) dt

"""

#See Kou papers for that function. (n)_j := n*(n+1)*(n+2)*...*(n+j-1)
# (n)_0=1
def shifted_fac(n,j): #ok
    res = mp.mpf(1)
    for k in range(j):
        res *= mp.mpf(n+k)
    return res


def H_minus1(a, b, c, n):    
    if mp.almosteq(a, 0, 1e-50): #case a = 0
        print("a=0 case for H-1")
        return factorial_arr[2*n]/ (factorial_arr[n]* mp.power(4*b,n)) * mp.sqrt(1/(2*b))#ok    
    
    else: #we only apply the formula for n<= 0 because it's the only one used
        if n >=0: #ok
            res = mp.mpf(0)
            for j in range(0, n+1):
                res += shifted_fac(-n,j)*shifted_fac(n+1,j)/(factorial_arr[j] * mp.power(-2 *mp.sqrt(2*mp.power(a,2)*b), j))
            res *= mp.exp(-a*c - mp.sqrt(2* mp.power(a,2)*b)) * mp.sqrt(1/(2*b)) * mp.power(mp.sqrt(mp.power(a,2)/(2*b)), n)            
            return res
        elif n<=-1:  #ok Fixed
            res = mp.mpf(0)
            for j in range(0, -n): #we want j to start from 0 till -n-1( >0) hence upper bound at -n
                res += shifted_fac(-n,j)*shifted_fac(n+1,j)/(factorial_arr[j] * mp.power(-2 *mp.sqrt(2*mp.power(a,2)*b), j))
            res *= mp.exp(-a*c - mp.sqrt(2* mp.power(a,2)*b)) * mp.sqrt(1/(2*b)) * mp.power(mp.sqrt(mp.power(a,2)/(2*b)), n)            
            return res
        
    
#See Kou&Wang paper : similarly a recurrence relation is found for Hi functions (under assumptions that are verified here)
def Hi_rec(a, b, c, n, m, spread = "product"):
    i_range = m+2
    n_range = n+m+3
    #Formatting : Hi_arr[i,n] = Hi(a,b,c,n)
    Hi_arr = mp.matrix(i_range, n_range) #starts at -1 for both i and n indexes
    for u in range(i_range):
        for v in range(n_range):
            Hi_arr[u,v] = mp.nan
    
    b_minus_c_square_half = b-mp.power(c,2)/2
    b_minus_c_pow = mp.matrix([0] * (n_range))
    b_minus_c_pow_div_fac = mp.matrix([0] * (n_range))
    b_minus_c_pow[0] = 1
    b_minus_c_pow_div_fac[0] = 1
    for k in range(1,n_range):
        b_minus_c_pow[k] = b_minus_c_square_half * b_minus_c_pow[k-1] #ok fixed
        b_minus_c_pow_div_fac[k] = b_minus_c_pow[k]/factorial_arr[k]  #ok

    def compute_Hi_rec(a, b, c, nn, i):

        if i == -1 and mp.isnan(Hi_arr[i+1, nn+1]) : 
            Hi_arr[i+1, nn+1] = H_minus1(a,b,c,nn) #edge cases: compute with regular formula     
        
        #see paper Lemma A.2.
        elif i == 0 and  mp.isnan(Hi_arr[i+1, nn+1]): 
            if mp.almosteq(b,mp.power(c,2)/2, 1e-50):
                Hi_arr[i+1, nn+1] =  c/(2*nn +mp.mpf(2)) * compute_Hi_rec(a, b, c, nn+1, -1) - a/(2*nn +mp.mpf(2))*compute_Hi_rec(a, b, c, nn, -1) #ok
            else:
                rec_term = factorial_arr[nn]/ b_minus_c_pow[nn+1]
                                
                if mp.almosteq(a, 0, 1e-50): # a= 0 #ok
                    print("case a =0")
                    temp =0 
                    for k in range(0,nn+1):
                        temp += b_minus_c_pow_div_fac[k]*compute_Hi_rec(a, b, c, k, -1)
                    Hi_arr[i+1, nn+1] = rec_term/2 - rec_term * c/2 * temp 
                    
                elif a >0: #ok
                    temp =0 
                    for k in range(0,nn+1):
                        temp += b_minus_c_pow_div_fac[k]*(a/2 * compute_Hi_rec(a, b, c, k-1, -1) - c/2 * compute_Hi_rec(a, b, c, k, -1))
                    Hi_arr[i+1, nn+1] =  rec_term * temp 
                    
                else: #a < 0 #ok
                    temp =0 
                    for k in range(0,nn+1):
                        temp += b_minus_c_pow_div_fac[k]*(a/2 * compute_Hi_rec(a, b, c, k-1, -1) - c/2 * compute_Hi_rec(a, b, c, k, -1))
                    Hi_arr[i+1, nn+1] =  rec_term + rec_term * temp 
                    
        elif mp.isnan(Hi_arr[i+1, nn+1]): #i > 0 #ok
            Hi_arr[i+1, nn+1] = mp.mpf(1/i) * compute_Hi_rec(a, b, c, nn+1, i-2) - mp.mpf(c/i) * compute_Hi_rec(a, b, c, nn+1, i-1) - mp.mpf(a/i) * compute_Hi_rec(a, b, c, nn, i-1) #compute recursively cf. Assumption 4.1
        
        return Hi_arr[i+1, nn+1]       
        
    for u in range(0,n+1): compute_Hi_rec(a,b,c,u,m) 
    #In practice, we may think that it only computes across n (so the horizontal)
    # but according to the recurrence formula it will compute all Hi(a,b,c,n) for i <= m till i = 0 hence the vertical      

    if spread == "horizontal":
        #This function returns all Hi(a,b,c,u) where u goes from 0 to n and i is fixed 
        #Define Hi_arr as H[i,n] = Hi(a,b,c,n)
        # The computation of the values will require 1st idx(i) to go from  from -1 to i
        # We leave some room for 2nd idx (n) because some calls will be made to beyond H_iw where w >n  
        return Hi_arr[m+1,1:n+2] #only interested in Hi for i fixed and nn from 0 to n (offset means idx 0 is i = -1)

    
    elif spread == "product":
        #This function returns all Hi(a,b,c,u) where u goes from 0 to n and i goes from 0 to m
        #Define Hi_arr as H[i,n] = Hi(a,b,c,n)
        # The computation of the values will require 1st idx (i) to go from  from -1 to m
        # We leave some room for 2nd idx (n) because some calls will be made to beyond H_iw where w >n
         return Hi_arr[1:,1:n+2] #akk Hu(a,b,c,v) where 0 <= u <= m and 0<= v <= n
    
    elif spread == "vertical":
        #not supposed to be called, so not implemented
        exit(0)

    else: #wrong argument
        print("Wrong argument for spread") 
        exit(-1)


def factorial(m): #ok
    global factorial_arr
    factorial_arr = mp.matrix([0] * (m+1)) #as binomial[k][n]    
    for n in range(m+1):
        factorial_arr[n] = mp.factorial(n)            
    
def binomial(m): #ok
    #Store 2d array of binomial coefficients
    global binomial_arr
    binomial_arr = mp.matrix(m+1) #as binomial[k][n]    
    for n in range(m+1):
        for k in range(0,n+1):
            binomial_arr[k,n] = mp.binomial(n,k)
            
def eta_powers(eta_1, eta_2, m): #ok
    global arr1, arr2           
    #Store array of powers of eta_1 / (eta_1 + eta_2) 
    u1 = eta_1 / (eta_1 + eta_2)
    arr1 = mp.matrix([mp.power(u1,k) for k in range(m+1)])

    #Store array of powers of eta_2 / (eta_1 + eta_2) 
    u2 = eta_2 / (eta_1 + eta_2)
    arr2 = mp.matrix([mp.power(u2,k) for k in range(m+1)])
    

def q_p_powers(p, q, m): #ok
    global p_pow, q_pow
    q = mp.mpf(1-p)
    p_pow = mp.matrix([mp.power(p,k) for k in range(m+1)])
    q_pow = mp.matrix([mp.power(q,k) for k in range(m+1)])
    
def P_and_Q(eta_1, eta_2, p, m): #ok
    global P_arr, Q_arr    
    q = mp.mpf(1-p)
    #Computation of P(n,k) and Q(n,k):
    P_arr = mp.matrix(m+1)
    Q_arr = mp.matrix(m+1)
    for n in range(m+1):
        for k in range(1,n+1):
            if k==n :
                P_arr[n,k] = mp.power(p,n)
                Q_arr[n,k] = mp.power(q,n)
            else:
                for i in range(k,n):
                    P_arr[n,k] +=  binomial_arr[i-k, n-k-1] * binomial_arr[i, n] * arr1[i-k]*arr2[n-i] * p_pow[i] * q_pow[n-i]  
                    Q_arr[n,k] +=  binomial_arr[i-k, n-k-1] * binomial_arr[i, n] * arr1[n-i]*arr2[i-k] * p_pow[n-i] * q_pow[i]  
    
    return P_arr, Q_arr


def P_tilde_and_Q_tilde(eta_1, eta_2, p, m): #ok fixed
    q = 1-p
    #Indexes letters used fit with the paper notations
    global P_tilde_arr, Q_tilde_arr
    P_tilde_arr = mp.matrix(m+1)
    Q_tilde_arr = mp.matrix(m+1)
    for n in range(m+1):
        for i in range(1,n+1):
            if i==1 :
                for j in range(1, n+1):
                    P_tilde_arr[n,1]+= Q_arr[n,j]*arr2[j]
                    Q_tilde_arr[n,1] += binomial_arr[j,n] * q_pow[j] * p_pow[n- j] * binomial_arr[j-i,n-i] * arr2[j-i] * arr1[n-j+1]
            else:
                P_tilde_arr[n,i] = P_arr[n,i-1]
                for j in range(i,n+1):
                    Q_tilde_arr[n,i] += binomial_arr[j,n] * q_pow[j] * p_pow[n- j] * binomial_arr[j-i,n-i] * arr2[j-i] * arr1[n-j+1]
    return P_tilde_arr, Q_tilde_arr
    
    

def lap_psi_tilde(alpha, mu, sigma, lambda_, p, eta_1, eta_2, a, b, T, m, debug = False):
    q = 1-p
    
    G = lambda x : x * mu + mp.power(x*sigma,2)/mp.mpf(2) + lambda_ * (p*eta_1/(eta_1-x) + q*eta_2/(eta_2 +x) - mp.mpf(1) )        
    G_derivative = lambda x : mu + x*sigma**2+ lambda_ * (p*eta_1/(eta_1-x)**2 - q*eta_2/(eta_2 +x)**2) #eventually for Newton's method

    try:
        beta_1 = mp.findroot(lambda x : G(x) - alpha, (1e-30,eta_1-1e-30), tol=mp.mpf(1e-50), solver = 'ridder')
        if beta_1 <0: raise Exception
    except:
        print("Failed to find root beta_1, switching solver")
        beta_1 = mp.findroot(lambda x : G(x) - alpha, (1e-30,eta_1-1e-30), tol=mp.mpf(1e-50), solver = 'anderson')
        print("beta_1:",beta_1)
        print("error_1:", G(beta_1) - alpha)
        
    try:
        beta_2 =  mp.findroot(lambda x : G(x) - alpha, (eta_1+1e-5, 1e4), tol=mp.mpf(1e-50), solver = 'ridder')
        if beta_2 <0: raise Exception
          
    except:
        print("Failed to find root beta_2, switching solver")
        beta_2 =  mp.findroot(lambda x : G(x) - alpha, (eta_1+1e-5, 150), tol=mp.mpf(1e-50), solver = 'anderson')
        print("beta_2:",beta_2)
        print("error_2:", G(beta_2) - alpha)

    
    if debug:
        print("alpha:", alpha)
        print("beta_1:",beta_1)
        print("error_1:", G(beta_1) - alpha)
        print("beta_2:",beta_2)
        print("error_2:", G(beta_2) - alpha)
    
    
    if beta_1 <0 or beta_2 <0 : print([beta_1, beta_2])
    if mp.almosteq(beta_1,beta_2, 1e-7): print("error: same root found twice")

    A = (eta_1 - beta_1) /(beta_2 - beta_1) * mp.exp(-b*beta_1) + (beta_2 - eta_1)/(beta_2 - beta_1) * mp.exp(-b*beta_2) #ok
    B = (eta_1 - beta_1)*(beta_2 - eta_1)/(eta_1 *(beta_2 - beta_1)) * (mp.exp(-b*beta_1) - mp.exp(-b*beta_2))    #ok
   
    c_plus = sigma * eta_1 + mu/sigma #ok
    c_minus = sigma * eta_2 - mu/sigma #ok
    upsilon = alpha + lambda_ + mp.power(mu,2)/(mp.mpf(2) * mp.power(sigma,2)) #ok
    h = (b-a)/sigma #ok
    
    #ok
    exp_lambda_DL = mp.matrix([0] * (m+1))
    exp_lambda_DL[0] = mp.mpf(1)
    for n in range(1,m+1): 
        exp_lambda_DL[n] =exp_lambda_DL[n-1]* lambda_ / mp.mpf(n) 
        
    #ok
    exp_lambda_p_DL = mp.matrix([0] * (m+1))
    exp_lambda_p_DL[0] = mp.mpf(1)
    for n in range(1,m+1):
        exp_lambda_p_DL[n] =exp_lambda_DL[n]* p_pow[n] 
        
    sigma_eta1_pow = mp.matrix([mp.power(sigma*eta_1,k) for k in range(m+1)]) #ok
    sigma_eta2_pow = mp.matrix([mp.power(sigma*eta_2,k) for k in range(m+1)]) #ok
        
    #See paper : The Laplace transform consists of 4 terms (1 per line), we code it the same way
    #Regarding the computations of Hi (highest computational complexity):
    # 1) H0(-h, upsilon, -mu/sigma, n) for n= 0 to n =m : we compute this once and don't keep it in memory : use Hi_rec() with horizontal spread
    # 2) Hi(h, upsilon, c+, n) for (i,n) in [0,m] x [1,m] : we compute it using Hi_rec() with product spread
    # 3) Hi(-h, upsilon, c-, n) for (i,n) in [0,m] x [1,m] : we compute it using Hi_rec() with product spread
    # 4) Hi(h, upsilon, c+, n) for (i,n) in [0,m] x [1,m]: re-use 2)
    # 5)H0(h, upsilon, c+, 0) : re-use 2) because Hi_rec_prod() actually computes from n=0
    
    #First compute and store the Hi mentioned above: ok
    H0_arr = Hi_rec(-h, upsilon, -mu/sigma, m, 0, spread="horizontal")
    Hi_t2_arr = Hi_rec(h, upsilon, c_plus, m, m, spread="product")
    Hi_t3_arr = Hi_rec(-h, upsilon, c_minus, m, m, spread="product")
   
    #term 1 : checked to be stable through variations of m
    #ok
    term1 = 0
    for n in range(0,m+1):
        term1 += exp_lambda_DL[n] * H0_arr[n]
    term1 = (A+B) * term1
    
    #term 2:checked to be stable through variations of m
    #ok
    term2 = 0    
    for n in range(1,m+1):
        term2_temp = 0
        for j in range(1,n+1):
            temp = 0
            for k in range(j):
                temp += sigma_eta1_pow[k] * Hi_t2_arr[k, n]
            term2_temp += (A*P_arr[n,j] + B * P_tilde_arr[n,j]) * temp
        term2 += exp_lambda_DL[n]*term2_temp
    
    term2 *= mp.exp(h*sigma*eta_1)
   
    #term 3:checked to be stable through variations of m
    #ok
    term3 = 0
    for n in range(1,m+1):
        term3_temp = 0
        for j in range(1,n+1):
            temp = 0
            for k in range(j):
                temp += sigma_eta2_pow[k] * Hi_t3_arr[k, n]
            term3_temp += (A*Q_arr[n,j] + B * Q_tilde_arr[n,j])*temp
        term3 += exp_lambda_DL[n]*term3_temp
    
    term3 *= -mp.exp(-h*sigma*eta_2)

    
    #term 4: checked to be stable through variations of m
    #ok
    term4 = 0
    for n in range(1,m+1):
        temp = 0
        for k in range(n+1):
            temp += sigma_eta1_pow[k] *Hi_t2_arr[k, n]
        term4+= exp_lambda_p_DL[n] *temp
    term4 += Hi_t2_arr[0, 0]
    term4 *= B*mp.exp(h*sigma*eta_1) 
    
    
    if debug:
        print("A:", A)
        print("B:", B)
        print("term1:",term1)
        print("term2:",term2)
        print("term3:",term3)    
        print("term4:",term4)           
            
    return term1 + term2 + term3 + term4


#Gaver-Stehfest algorithm to compute inverse Laplace transform
def gaver_stehfest(f_lap, t, w, richardson=False):
    #This uses n-point Richardson extrapolation for faster convergence
    
    burning_out = 3
    ones_arr = mp.matrix([mp.power(mp.mpf(-1),k) for k in range(w+1+burning_out)])

    
    #Keep in memory the values of the Laplace transform of singular values as they will be used multiple times
    if richardson: 
        f_lap_arr = mp.matrix([f_lap(k * mp.log(2)/t) for k in tqdm(range(1, 2*(w+burning_out) +1))])
    else: f_lap_arr = mp.matrix([f_lap(k * mp.log(2)/t) for k in tqdm(range(1,2*w+1))])
    
    
    def f_tilde_n(f_lap, t, n): #ok
        res = mp.mpf(0)
        if n == 0 : print("n=0!")
        for k in range(n+1):
            res += ones_arr[k] * binomial_arr[k,n] * f_lap_arr[k+n-1] #did not compute f_lap for n=0 so there is an offset of 1
        return mp.log(2)/t * binomial_arr[n, 2*n] * n * res
        
    
    if richardson:        
        f_tilde_arr = mp.matrix([f_tilde_n(f_lap, t, k) for k in range (1+burning_out, w+1+burning_out)])
        res = mp.mpf(0)
        for k in range(1,w+1):
            res += omega_arr[k]*f_tilde_arr[k-1]
        return res
    

    return f_tilde_n(f_lap, t, w)
    
    

#########################################################################################

def european_barrier_uic_option(S0, r, sigma, lambda_, p, eta_1, eta_2, K, H, T, m, w, richardson= True):
    
    #w : number of points for the inverse Laplace transform
    q = 1-p
    zeta = p*eta_1/(eta_1 - 1) + q*eta_2/(eta_2 +1) - 1
    mu =  r - mp.power(sigma,2)/2 - lambda_ *zeta    
    a = mp.log(K/S0) 
    b = mp.log(H/S0)
    
    p_tilde = p/(1+zeta) *eta_1/(eta_1 - 1)
    mu_tilde= r +  mp.power(sigma,2)/2 - lambda_*zeta
    eta_1_tilde = eta_1 -1
    eta_2_tilde = eta_2 +1
    lambda_tilde = lambda_*(zeta + 1)
    
    if richardson: 
        factorial_arr = mp.matrix([mp.factorial(k) for k in range(w+1)])
        global omega_arr
        omega_arr = mp.matrix([0] * (w + 1)) 
        for k in range(w+1):
            omega_arr[k] = mp.power(-1,w-k) * mp.power(k,w)/(factorial_arr[k] * factorial_arr[w-k])      


    eta_powers(eta_1_tilde, eta_2_tilde, m)
    q_p_powers(p_tilde, 1-p_tilde, m) 
    P_and_Q(eta_1_tilde, eta_2_tilde, p_tilde, m)
    P_tilde_and_Q_tilde(eta_1_tilde, eta_2_tilde, p_tilde, m)    
    laplace_transform_t1= lambda alpha : lap_psi_tilde(alpha, mu_tilde, sigma, lambda_tilde, p_tilde, eta_1_tilde, eta_2_tilde, a, b, T, m, debug = True)
    t1 = gaver_stehfest(laplace_transform_t1, b, w, richardson)
    
    eta_powers(eta_1, eta_2, m)
    q_p_powers(p, 1-p, m) 
    P_and_Q(eta_1, eta_2, p, m)
    P_tilde_and_Q_tilde(eta_1, eta_2, p, m)
    laplace_transform_t2= lambda alpha : lap_psi_tilde(alpha, mu, sigma, lambda_, p, eta_1, eta_2, a, b, T, m, debug = False)
    t2 =  gaver_stehfest(laplace_transform_t2, b, w, richardson) 
    
    return   t1 - K*mp.exp(-r*T)*t2
    

def toy_example_paper():
    b = mp.mpf(0.3)
    a = mp.mpf(0.2)
    T = mp.mpf(1.0)
    mu = mp.mpf(0.1)
    sigma = mp.mpf(0.2)
    p = mp.mpf(0.5)
    eta_1 = mp.mpf(1/0.02)
    eta_2 = mp.mpf(1/0.03)
    llambda = mp.mpf(3.0)
    m = 15 #number of points before truncation of the infinite sums 
    w = 20 #number of points for laplace inverse trasnform 
    
    binomial(2*max(w+3,m+3)) #account for the burning out
    factorial(3*max(w+3,m+3))
    
    richardson = True 
    
    if richardson:
        factorial_arr = mp.matrix([mp.factorial(k) for k in range(w+1)])
        global omega_arr
        omega_arr = mp.matrix([0] * (w + 1)) 
        for k in range(w+1):
            omega_arr[k] = mp.power(mp.mpf(-1),w-k) * mp.power(k,w)/(factorial_arr[k] * factorial_arr[w-k])      #ok
        
    eta_powers(eta_1, eta_2, m)
    q_p_powers(p, 1-p, m) 
    P_and_Q(eta_1, eta_2, p, m)
    P_tilde_and_Q_tilde(eta_1, eta_2, p, m)
    laplace_transform = lambda alpha : lap_psi_tilde(alpha, mu, sigma, llambda, p, eta_1, eta_2, a, b, T, m, debug = False)
    print( gaver_stehfest(laplace_transform, b, w, True))
    

def testing_H(a,b,c,n,i):
    #Hh_n(x) = 1/n! * int_x^\infty (t-x)^n *exp(-t^2/2) dt
    #H_i(a,b,c,n) = 1/sqrt(2.pi) * int_0^\infty exp[(0.5*c^2 -b)*t] * t^(n+i/2) Hh_i(c*sqrt(t) + a /sqrt(t)) dt
    
    #Notes on experimentation:
    #Hi = Hi_rec = 0.232233047 for a=0, b=1, c=1, n=1, i=0
    #Hi = Hi_rec = 17.977362252 for a=0, b=1, c=1, n=5, i=0  
    #Hi = Hi_rec = 6.844050650 for a=0, b=1, c=1, n=5, i=3 
    #Hi = Hi_rec = 0.000000558759 for a=0, b=2.5, c=3.8, n=4, i=7 
    #Hi= Hi_rec = 0.403496643773852 for a=1, b=1, c=1, n=3, i=-1
    #Hi= Hi_rec = 2.981459336575 for a=-1, b=1, c=1, n=3, i=-1
    #Hi = Hi_rec for a!=0 (whether a <0 or a >0) if i!= -1 (now that's it's fixed)
    #Note that for high values of i and n, we stumble upon small differences, that are likely to be caused by the nested in quadrature computation
    
    binomial(40) #account for the burning out
    factorial(40)
    
    if i ==-1: hh_i = lambda x : mp.exp(-mp.power(x,2)/2)
    else: hh_i = lambda x : 1/mp.factorial(i)*mp.quad(lambda t : mp.power(t-x, i) * mp.exp(-mp.power(t,2)/mp.mpf(2)), [x, mp.inf])
    Hi, err =   mp.quad(lambda t : mp.exp((mp.power(c,2)/mp.mpf(2)-b)*t) * mp.power(t, n+i/2) * hh_i(c*mp.sqrt(t) + a /mp.sqrt(t)), [0, mp.inf],error=True)
    Hi *=1/mp.sqrt(2*mp.pi) 
    
    print("a ={}, b={}, c={}, n={}, i={}".format(a,b,c,n,i))
    print("Analytic Hi:", Hi)
    print("quad err:", err)
    
    if i==-1:
        print(H_minus1(a, b, c, n))
    else:
        print("Recursive Hi:", Hi_rec(a, b, c, n, i)[i,n])

    
if __name__ == '__main__':
    """
    # Parameters in Kou's paper: Option Pricing Under a Double Exponential Jump Diffusion Model under Laplace inverse and numerical examples section    
    eta_1 = mp.mpf(1/0.02)
    eta_2 = mp.mpf(1/0.04)
    p = mp.mpf(0.3)
    lambda_ = mp.mpf(3.0)
    r = mp.mpf(0.05)
    sigma = mp.mpf(0.2)
    K = mp.mpf(100)
    H = mp.mpf(120)
    T= mp.mpf(1.0)
    S0 = mp.mpf(100)    

    m = 15 #number of points before truncation of the infinite sums 
    w = 10 #number of points for laplace inverse trasnform 

    #Compute and store re-used values 
    binomial(2*max(w+3,m+3)) #account for the burning out
    factorial(3*max(w+3,m+3))
    
    print("Price of the UIC barrier option:", european_barrier_uic_option( S0, r, sigma, lambda_, p, eta_1, eta_2, K, H, T, m, w)) 
    """
    toy_example_paper()
    #testing_H(mp.mpf(-3),mp.mpf(1),mp.mpf(5),2,3)

    
    




#The lord opposes the proud and here I am to humble you
#Heavenly father bear witness to the strength of your creation
   
