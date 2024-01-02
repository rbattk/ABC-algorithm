# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 14:25:34 2021

@author: user1
"""

import random
import pandas as pd
from sklearn import preprocessing
import numpy as np

dim=8; #There are 8 passive components
# for E-24 #####
e24=[0.1, 0.11, 0.12, 0.13, 0.15, 0.16, 0.18, 0.20, 0.22, 0.24, 0.27, 0.30, 
     0.33, 0.36, 0.39, 0.43, 0.47, 0.51, 0.56, 0.62, 0.68, 0.75, 0.82, 0.91]
low = 0.1;
upper = 0.91;
###############
# for E-96 #####
#e96=[0.1, 0.102, 0.105, 0.107, 0.110, 0.113, 0.115, 0.118, 0.121,
#0.124, 0.127, 0.130, 0.133, 0.137, 0.140, 0.143, 0.147, 0.150, 0.154, 
#0.158, 0.162, 0.165, 0.169, 0.174, 0.178, 0.182, 0.187, 0.191, 0.196, 
#0.2, 0.205, 0.210, 0.215, 0.221, 0.226, 0.232, 0.237, 0.243, 0.249, 
#0.255, 0.261, 0.267, 0.274, 0.280, 0.287, 0.294, 0.301, 0.309, 0.316, 
#0.324, 0.332, 0.340, 0.348, 0.357, 0.365, 0.374, 0.383, 0.392, 0.402, 
#0.412, 0.422, 0.432, 0.442, 0.453, 0.464, 0.475, 0.487, 0.499, 0.511, 
#0.523, 0.536, 0.549, 0.562, 0.576, 0.590, 0.604, 0.619, 0.634, 0.649, 
#0.665, 0.681, 0.698, 0.715, 0.732, 0.750, 0.768, 0.787, 0.806, 0.825, 
#0.845, 0.866, 0.887, 0.909, 0.931, 0.953, 0.976]
#low = 0.1;
#upper = 0.976;"""
###############
#butter#E-12 için#
#e12=[0.1, 0.12, 0.15, 0.18,0.22, 0.27, 0.33, 0.39, 0.47, 0.56, 0.68, 0.82]
#low=0.1;
#upper=0.82;
###############
bound=[(low,upper)];#passive component limit
pop=50;
N=pop/2;#number of food sources
limit=N*dim;
iters=50000;
R = []
C = []
z = []
"""Nnew=(N[i][j] + fi*(N[i][j] - r[j]))"""
#fi = random.uniform(-1,1);
#fi2 = random.uniform(0,1);        
"The common p value is randomly selected as i!=p. Let p = 0.1."
p=0.1;
"r is chosen as a random decision variable"
"There are newly created randomly selected variables."
"These are fitness values."
t1 = [];
fark=[];
"Objective Function"

"""def butter(x):
    Wc1 = 1/((x[0]*x[1]*x[4]*x[5])**0.5)
    Wc2 = 1/((x[2]*x[3]*x[6]*x[7])**0.5)
    Q1 = ((x[0]*x[2]*x[4]*x[5])**0.5)/((x[0]*x[4]) + (x[1]*x[4]))
    Q2 = ((x[2]*x[3]*x[6]*x[7])**0.5)/((x[2]*x[6]) + (x[3]*x[6]))
    Wc = 10
    Qt1 = 1.3065
    Qt2 = 0.5412
    Err_w_BF = (abs(Wc1 - Wc) + abs(Wc2 - Wc))/Wc
    Err_q_BF = (abs(Qt1 - Q1)/Qt1) + (abs(Qt2 - Q2)/Qt2) 
    k = 0
    for i in range(len(x)):
        k += (0.5*Err_w_BF + 0.5*Err_q_BF);
        
    return k      
    #return  (0.5*Err_w_BF + 0.5*Err_q_BF)"""
    
def state(x):
    Wsvf = ((x[3]/x[2])*(1/(x[6]*x[7]*x[4]*x[5])))**0.5
    Qsvf = ((x[2]*(x[0]+x[1]))/(x[0]*(x[2]+x[3])))*(((x[6]*x[3]*x[4])/(x[7]*x[2]*x[5]))**0.5)
    Wt = 10
    Qt = 0.707
    Err_w_SVF = (abs(Wsvf - Wt))/Wt
    Err_q_SVF = (abs(Qsvf - Qt))/Qt
    k = 0
    for i in range(len(x)):
        k += (0.5*Err_w_SVF + 0.5*Err_q_SVF);
        
    return k 
    
"Working bee phase of the algorithm"
def employed(N, f, trial, p=0.1):

    for i in range(len(N)):
        
        fit = []
        R  = N.copy()
        R.remove(N[i])
        r = random.choice(R)
        
        for j in range(len(N[0])):   
            
             fit.append( (N[i][j] + random.uniform(-1,1)*(N[i][j] - r[j])) )
                                                          
        "border control"
                                                          
        if f(N[i]) > f(fit):
            N[i] = fit
            trial[i] = 0
            
        else:
            trial[i] = trial[i] + 1
            
    
    return N, trial
"Probability calculus"
def P(N, f):
    
    Prob = []
    sP = sum ([1 / (1 + f(i) ) for i in N])
    for i in range(len(N)):
        
        Prob.append(  (1 / (1 + f(N[i]) ) )/  sP )
        
    return Prob

"Algorithm's tracker bee stage"
def onlooker(N, f, trial, p=0.1):
    
    Pi  =  P(N, f)
 
    for i in range(len(N)):

        if random.random() < Pi[i]:
            
            fit = []
            R  = N.copy()
            R.remove(N[i])
            r = random.choice(R)
            
            "Since we did not initially produce a new solution set, N[0]"
            for j in range(len(N[0])):  
                
                fit.append ( (N[i][j] + random.uniform(-1,1)*(N[i][j] - r[j])) )
            
            "border control"
                                                          
            if f(N[i]) > f(fit):
               N[i] = fit
               trial[i] = 0
            
            else:
              trial[i] = trial[i] + 1
   
    return N, trial
"Algorithm's tracker bee stage"
"Only solutions with trials>limit can proceed to the discovery phase."
def scout(N, trial, bound, limit):
    
    
    for i in range(len(N)):
    
        if trial[i] > limit : 
            trial[i] = 0
            N[i] = [bound[i][0] + ( random.uniform(0,1)*(bound[i][1] - bound[i][0]) ) for i in range(len(N[0]))]
        
    return N
"ABC algorithm"
def ABC(dim, bound, f, limit, pop, iters ):
    
    #bound = [(0.1, 0.976) for i in range(dim)]                
    bound = [(0.1, 0.91) for i in range(dim)]  
    #bound = [(0.1, 0.82) for i in range(dim)] 
    N = [[bound[i][0] + ( random.uniform(0,1)*(bound[i][1] - bound[i][0]) ) for i in range(dim)] for i in range(pop)]

    trial = [0 for i in range(pop)]
    
    
    while iters > 0:
        
        N, trial= employed(N, f, trial)
        
        N, trial= onlooker (N, f, trial)
        
        N = scout(N, trial, bound, limit)
        
        iters = iters - 1
    
    for i in N:    
        fx = [f(i)]
        I = fx.index(min(fx))  
        
    #print(N[I])
    z.append(N[I])
       
    
    for k in  range(4):
        R.append(z[0][k])
        C.append(z[0][k+4])
    df = pd.DataFrame(
    {"R" : R,
    "C" : C,})
    df = df.astype(float)
    scaler = preprocessing.MinMaxScaler(feature_range = (2,4))
    t=scaler.fit_transform((df))
    
    for l in range(2):
        for k in range(4):
            b = t[k][l]
            t1.append(round(b))    
    print("a1,b1,c1,d1,e1,f1,g1,h1:",t1);
    
    for u in range(dim):
        for y in range(len(e24)):           
            s = z[0][u]-e24[y];    
            fark.append(abs(s))
        z[0][u] = e24[np.argmin(fark)];
        fark.clear()
    print("a,b,c,d,e,f,g,h:",z)
            
            
    R1 = z[0][0]*100*(10**(t1[0]))/1000;
    R2 = z[0][1]*100*(10**(t1[1]))/1000;
    R3 = z[0][2]*100*(10**(t1[2]))/1000;
    R4 = z[0][3]*100*(10**(t1[3]))/1000;
    R5 = z[0][4]*100*(10**(t1[4]))/1000;
    R6 = z[0][5]*100*(10**(t1[5]))/1000;
    C1 = z[0][6]*100*(10**(t1[6]))/1000;
    C2 = z[0][7]*100*(10**(t1[7]))/1000;
    
    print("R1:",R1,"R2:",R2,"R3:",R3,"R4:",R4,
          "R5:",R5,"R6:",R6,"C1:",C1,"C2:",C2)
    
    return min(fx)  
