from chi_fct_kou import european_call_option
from psi_fct_kou import european_barrier_uic_option, european_barrier_uoc_option

import numpy as np
import mpmath as mp
import argparse


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument("--option", type=str, help="The type of option to price. Choose between [european, barrier-uic, barrier-uoc].", required = True)
    parser.add_argument("--eta_1", type=float, help="See Kou paper", default = 10.0 ,required = False)
    parser.add_argument("--eta_2", type=float, help="See Kou paper", default = 5.0 ,required = False)    
    parser.add_argument("--llambda", type=float, help="See Kou paper", default=1.0, required=False)    
    parser.add_argument("--p", type=float, help="See Kou paper", default=0.4, required = False)
    parser.add_argument("--r", type=float, help="risk-free interest rate", default=0.05, required=False)
    parser.add_argument("--sigma", type=float, help="volatility", default=0.16, required=False)
    parser.add_argument("--K", type=float, help="Strike price of the option", default=98, required=False)
    parser.add_argument("--T", type=float, help="Maturity (in years)", default=0.5, required=False)    
    parser.add_argument("--S0", type=float, help="Initial price of the option", default=100, required=False)
    parser.add_argument("--barrier", type=float, help="Value of the barrier for the barrier option", default=130, required=False)
    parser.add_argument("--m", type=int, help="Number of points before truncation of sums", default=15, required=False)
    parser.add_argument("--w", type=int, help="Number of points in Gaver-Stehfest inverse Laplace transform", default=10, required=False)
    args = parser.parse_args()   
    

    if args.option == "european":         
        print("Price of the european call option: {:.4f}$".format(european_call_option(args.S0, args.r, args.sigma, args.llambda, args.p, args.eta_1, args.eta_2, args.K, args.T, args.m)))
        
    elif args.option == "barrier-uic":
        print("Price of the UIC barrier option: {:.4f}$".format(european_barrier_uic_option(mp.mpf(args.S0), mp.mpf(args.r), mp.mpf(args.sigma), mp.mpf(args.llambda), mp.mpf(args.p), mp.mpf(args.eta_1), mp.mpf(args.eta_2), mp.mpf(args.K), mp.mpf(args.barrier), mp.mpf(args.T), args.m, args.w))) 
    
    elif args.option == "barrier-uoc":
        print("Price of the UIC barrier option: {:.4f}$".format(european_barrier_uoc_option(mp.mpf(args.S0), mp.mpf(args.r), mp.mpf(args.sigma), mp.mpf(args.llambda), mp.mpf(args.p), mp.mpf(args.eta_1), mp.mpf(args.eta_2), mp.mpf(args.K), mp.mpf(args.barrier), mp.mpf(args.T), args.m, args.w)))
    
             
    else : raise Exception("There is no pricing model for the {} option".format(args.option))
   
