import numpy as np
import copy

def soft_threshold(x,sigma):
    return np.sign(x)*np.maximum(abs(x)-sigma,0)

def multiZ(A,rho,lmda1,lmda2):
    trueA = copy.copy(A)
    p = A[0].shape[0]
    K = len(A)
    X = [A[0]*0 for k in range(K)]
    fusions = np.zeros((K,K,p,p)) 
    newc = list()
    for k in range(K):
        others = list(range(K))
        others.pop(k)
        newc.append(A[0]*0)
        for o in others:
            newc[k] = newc[k] + (A[o]-A[k] < -1e-4) - (A[o]-A[k] > 1e-4)
    for iter in range(K-1):
        ordermats = []
        for k in range(K):
            others = list(range(K))
            others.pop(k)
            ordermats.append(A[0]*0)
            for o in others:
                ordermats[k] = ordermats[k] + (A[k]-A[o] > 1e-4)
            if k > 0:
                for o in range(k):
                    ordermats[k] = ordermats[k] + (abs(A[o]-A[k]) < 1e-4)
            ordermats[k] = ordermats[k] + 1
        betas_g = []
        for k in range(K):
            betas_g.append(A[k]- lmda2/rho * newc[k])
        new_ordermats = []
        for k in range(K):
            others = list(range(K))
            others.pop(k)
            new_ordermats.append(A[0]*0)
            for o in others:
                new_ordermats[k] = new_ordermats[k] + ((betas_g[k] - betas_g[o]) > 1e-4)
            if k > 0:
                for o in range(k):
                    new_ordermats[k] = new_ordermats[k] + (abs(betas_g[o] - betas_g[k]) < 1e-4)
            new_ordermats[k] = new_ordermats[k] + 1
        for k in range(K):
            for kp in range(K):
                fusions[k][kp] = fusions[k][kp] + ((ordermats[k]-1 == ordermats[kp]) & (new_ordermats[k] < new_ordermats[kp])) \
                        + ((ordermats[k]+1 == ordermats[kp]) & (new_ordermats[k] > new_ordermats[kp])) \
                        + (abs(A[k] - A[kp]) < 1e-4)
        fusions = (fusions > 0) * 1
        for k in range(K):
            for kp in range(K):
                others = set(range(K)).difference({k}.union({kp}))
                for o in others:
                    bothfused = fusions[k][o] & fusions[kp][o]
                    fusions[k][kp] = fusions[k][kp] | bothfused
        for k in range(K):
            others = list(range(K))
            others.pop(k)
            fusemean = trueA[k]
            denom = A[0]*0 + 1
            for o in others:
                fusemean = fusemean + fusions[k][o] * trueA[o]
                denom = denom + fusions[k][o]
            A[k] = fusemean/denom
        newc = []
        for k in range(K):
            others = list(range(K))
            others.pop(k)
            newc.append(A[0]*0)
            for o in others:
                newc[k] = newc[k] + (A[o] - A[k] < -1e-4) - (A[o] - A[k] > 1e-4)
    for k in range(K):
        betas_g[k] = A[k] - lmda2/rho * newc[k]
    for k in range(K):
        X[k] = soft_threshold(betas_g[k], lmda1/rho)
    return X
