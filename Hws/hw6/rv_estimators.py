'''
#--------------------------------------------------------------------------
#   Auxiliary functions: realized variance measures
#   Translated to R from Roel Oomen's 01-Feb-2007 Matlab code on 6-Aug-2008
#   Translated to python from Jim Gatheral's 6-Aug-2008 R code on 9-Jan-2021
#--------------------------------------------------------------------------
'''

import numpy as np

'gamma function (Optimized by Jing Chen, class of 2012. )'

def gamma(p, h, q, k):
    '''
    Inputs:
    p = price series
    h = index of a given subsample
    q = subsampling frequency
    k = time-offset
    
    Output:
    the kth realized autocovariance of the subsampled series
    '''
    p = np.array(p) # convert p into a numpy array object
    n = len(p)
    
    if k==0:
        sample = p[np.arange(h, n+1, q) - 1]
        r = np.diff(sample)
        g = sum(r*r)
    elif q==1:
        diff_overlap = p[(h+k):(n-k)] - p[(h+k-1):(n-k-1)]
        r1 = np.concatenate([p[h:(h+k)] - p[(h-1):(h+k-1)], diff_overlap])
        r2 = np.concatenate([diff_overlap, p[(n-k):n] - p[(n-k-1):(n-1)]])
        g = sum(r1*r2) 
    else:
        index_m = np.arange(h + k*q, n - k*q + 1, q) - 1 #index of the overlapping part
        index_h = np.arange(h, h + k*q + 1, q) - 1 #index of the front
        index_t = np.arange(index_m[-1], n, q) #index of the tail
        nh = len(index_h)
        nt = len(index_t)
        
        diff_m = p[index_m[1:]] - p[index_m[:-1]]
        diff_h = p[index_h[1:]] - p[index_h[:-1]]
        diff_t = p[index_t[1:]] - p[index_t[:-1]]        
        r1 = np.concatenate([diff_h, diff_m])
        r2 = np.concatenate([diff_m, diff_t])
        g = sum(r1*r2)
    
    return g
    

# realized variance
def rv_plain(p, q):
    q = max(1, round(q))
    M = len(p) - 1
    rv = (M/q)/np.floor(M/q)*gamma(p, 1, q, 0)
    return rv

# Zhou's subsampling RV
def zhou(p, q):
    q = max(1, round(q))
    M = len(p) - 1 
    rv = 0
    for i in range(q):
        rv = rv + gamma(p, i+1, q, 0) + 2*gamma(p, i+1, q, 1)
    rv = M/(q*(M - q + 1))*rv
    return rv


# ZMA two-scale RV
def tsrv(p, q):
    q = max(2, round(q))
    if len(p) < q:
        print(f'length of p = {len(p)} is less than q = {q}.')
        return None
    ss = 0 
    rv = gamma(p, 1, 1, 0)
    M = len(p) - 1 
    Mb = (M - q + 1)/q
    for i in range(q):
        ss = ss + gamma(p, i+1, q, 0)
    rv = (ss/q - Mb/M*rv)/(1 - Mb/M)
    return rv

# Zhang's multiscale RV
def msrv(p, q):
    q = max(2, round(q))
    if len(p) < q:
        print(f'length of p = {len(p)} is less than q = {q}.')
        return None
    rv = 0
    i = np.arange(1, q+1)
    a = (12*(i/q**2)*(i/q - 1/2) - 6*i/q**3)/(1 - q**(-2))
    for h in range(q):
        ss = 0
        for j in range(h+1):
            ss += gamma(p, j + 1, h + 1, 0)/(h + 1)
        rv += a[h]*ss
    
    return rv

# Realized kernel with mod. Tukey-Hanning kernel
def krvth(p, q):
    p = np.array(p)
    q = max(1, round(q))
    if len(p) < q:
        print(f'length of p = {len(p)} is less than q = {q}.')
        return None
    r = np.diff(p)
    rv = gamma(p, 1, 1, 0)
    for s in range(q):
        x = s/q
        rv += 2*(np.sin(np.pi/2*(1-x)**2))**2*gamma(p, 1, 1, s + 1)
    
    return rv


# Realized kernel with Cubic kernel
def krvc(p, q):
    q = max(1, round(q))
    if len(p) < q:
        print(f'length of p = {len(p)} is less than q = {q}.')
        return None
    r = np.diff(p)
    rv = gamma(p, 1, 1, 0)
    for s in range(q):
        x = s/q
        rv += 2*(1 - 3*x**2 + 2*x**3)*gamma(p, 1, 1, s + 1)
    
    return rv

# Realized kernel with Parzen kernel
def krvp(p, q):
    q = max(1, round(q))
    if len(p) < q:
        print(f'length of p = {len(p)} is less than q = {q}.')
        return None
    r = np.diff(p)
    rv = gamma(p, 1, 1, 0)
    for s in range(q):
        x = s/q
        if x < 1/2:
            rv += 2*(1 - 6*x**2 + 6*x**3)*gamma(p, 1, 1, s + 1)
        else:
            rv += 2*2*(1 - x**3)*gamma(p, 1, 1, s + 1)
    
    return rv
