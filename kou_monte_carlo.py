import numpy as np
import matplotlib.pyplot as plt
"""
This file aims at determining the price of the options considered in Kou's paper using his double exponential jump diffusion model.
After studying the results, they are overall close to the values presented in Kou's paper (no strong deviation).
"""
rng = np.random.default_rng(19)

def monte_carlo_european(M=100000):
    # Parameters in Kou's paper from 2002 under European's options pricing section
    eta_1 = 10
    eta_2 = 5
    lambda_ = 1.0
    p = 0.4
    r = 0.05
    sigma = 0.16
    K = 98
    T = 0.5
    S0 = 100
    
    q = 1-p
    zeta = p*eta_1/(eta_1 - 1) + q*eta_2/(eta_2 +1) - 1
    mu =  r - sigma**2/2 - lambda_ *zeta   
    
    def exp_jump():
        return rng.exponential(1 / eta_1) if np.random.rand() < p else -rng.exponential(1 / eta_2)        

    Z = rng.normal(0,1, M)
    N = rng.poisson(lambda_ * T, M)
    jumps = np.array([sum(exp_jump() for _ in range(n)) for n in N])
    stocks = S0 * np.exp((mu-0.5*sigma**2)*T + sigma * np.sqrt(T) * Z + jumps) 
    
    maturity = np.where(stocks > K, stocks-K, 0)
    expected_maturity = np.average(maturity)
    print("Simulated value of the European option:", expected_maturity)   
    

def monte_carlo_barrier(M=30000):
    #M is the number of Monte Carlo simulations
    
    #Market parameters: 
    eta_1 = 1/0.02
    eta_2 = 1/0.04
    p=0.3
    lambda_ = 3.0
    r=0.05
    
    sigma=0.2
    K = 100
    H = 120
    T=1.0
    S0 = 100
    
    q = 1-p
    zeta = p*eta_1/(eta_1 - 1) + q*eta_2/(eta_2 +1) - 1
    mu =  r - sigma**2/2 - lambda_ *zeta    
    
    dt = 0.001 #time subdivisions
    n_steps = int(T/dt)
    
    def exp_jump():
        return rng.exponential(1 / eta_1) if np.random.rand() < p else -rng.exponential(1 / eta_2)
    
    paths = np.zeros((M, n_steps + 1))
    paths[:, 0] = S0

    for i in range(n_steps):
        Z = rng.normal(0, 1, M)                    # Brownian motion
        N = rng.poisson(lambda_ * dt, M)       # Number of jumps
        jumps = np.array([sum(exp_jump() for _ in range(n)) for n in N])  # Sum of jumps
        paths[:, i + 1] = paths[:, i] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z + jumps)
    
    """
    plt.plot(paths.T, alpha=0.1, color='blue')
    plt.title('Monte Carlo Simulation of Kou Model')
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.show()
    """
    
    maturity = np.zeros(M)
    for i in range(M):
        if np.any(paths[i]>H): #the uic barrier option is only usable if it crossed H 
            maturity[i] = paths[i,-1]-S0 if paths[i,-1] > S0 else 0
    

    expected_maturity = np.average(maturity)
    print("Simulated value of the barrier option:", expected_maturity)
    


if __name__ == '__main__':

    monte_carlo_european()
    monte_carlo_barrier()
    
    
    
    
