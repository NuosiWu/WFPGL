import numpy as np
cimport numpy as np
cimport cython
from numpy import linalg as LA

cpdef theta_updating(double[:,:] S, double[:,:] Z, double[:,:] V, double[:,:] shift, double[:,:] Theta, double nk, double rho = 1):
    p = S.shape[0]
    D = np.zeros(p, dtype=np.float64)
    tmp = np.zeros((p,p), dtype=np.float64)
    Dhat = np.zeros(p, dtype=np.float64)
    final = np.zeros((p,p), dtype=np.float64)
    for i in range(p):
        for j in range(p):
            tmp[i][j] = S[i][j] + rho/nk*(shift[i][j] - Z[i][j] + V[i][j])
    D,Vec = np.linalg.eig(tmp)
    for i in range(p):
        Dhat[i] = nk/2/rho*(-D[i]+np.sqrt(D[i]*D[i]+4*rho/nk))
    final = np.dot(Vec, np.dot(np.diag(Dhat),Vec.T))
    for i in range(p):
        for j in range(p):
            Theta[i][j] = final[i][j]
    return final

    
