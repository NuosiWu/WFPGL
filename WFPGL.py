from numpy import linalg as LA
import numpy as np
import pandas as pd
import pglasso
import updateZ
import copy

#def soft_threshold(x,sigma):
#    return np.sign(x)*max(abs(x)-sigma,0)

class WFPGL:
    def __init__(self,Y,pathways,lambda1,lambda2,prior,penalty="fused",mu=1,weights="equal",maxiter=5000,tol=1e-5,truncate=1e-5):
        self.Y = Y
        self.pathways = pathways
        self.penalty = penalty
        self.lamd1 = lambda1*1.0
        self.lamd2 = lambda2*1.0
        self.mu = mu*1.0
        self.weights = weights
        self.maxiter = maxiter
        self.tol = tol
        self.truncate = truncate
        self.k = len(Y) #k views
        tmp = [x.shape[1] for x in Y] #n_k
        self.nk = [x*1.0/sum(tmp) for x in tmp]
        #self.nk = [1.0 for x in range(len(Y))]
        self.W = prior
        # checking
        tmp = [x.shape[0] for x in Y]
        assert all(x==tmp[0] for x in tmp),"Each view must have same dimension!"
        self.p = tmp[0]
        self.S = [np.cov(x,bias=True) for x in self.Y]


    def initializing(self):
        #self.THETA0 = [np.eye(self.p) for x in range(self.k)]
        self.THETA1 = [np.eye(self.p) for x in range(self.k)]
        #self.U0 = [np.zeros((self.p,self.p)) for x in range(self.k)]
        self.U1 = [np.zeros((self.p,self.p)) for x in range(self.k)]
        #self.Z0 = [np.zeros((self.p,self.p)) for x in range(self.k)]
        self.Z1 = [np.zeros((self.p,self.p)) for x in range(self.k)]


    def fused_lasso(self,i,j):
        A1 = self.A[0][i][j]
        A2 = self.A[1][i][j]
        tmp = self.W[i][j]*self.lamd2/self.mu
        if A1 > (A2 + 2*tmp):
            return A1-tmp, A2+tmp
        elif A2 > (A1 + 2*tmp):
            return A1+tmp, A2-tmp
        else:
            return 0.5*(A1+A2), 0.5*(A1+A2)

    def fused_lasso_mat(self,A1,A2,para):
        tmpDif = A1-A2
        tmpSum = A1+A2
        Flag1 = tmpDif > 2*para
        Flag2 = tmpDif < -2*para 
        Flag3 = ~(Flag1 | Flag2)
        return Flag1*(A1-para)+Flag2*(A1+para)+Flag3*0.5*(tmpSum), Flag1*(A2+para)+Flag2*(A2-para)+Flag3*0.5*(tmpSum)


    def update_thetas(self):
        self.THETA0 = copy.copy(self.THETA1)
        for i in range(self.k):
            self.THETA1[i] = pglasso.pglasso(self.S[i], self.Z1[i], self.U1[i], self.pathways, self.mu, self.nk[i])

    def update_zs(self):
        #self.Z0 = copy.copy(self.Z1)
        self.A = map(lambda x,y: x+y, self.THETA1, self.U1)
        if self.k == 2:
            para = self.W*self.lamd2/self.mu
            self.Z1[0], self.Z1[1] = self.fused_lasso_mat(self.A[0],self.A[1],para)
            self.Z1[0] = np.sign(self.Z1[0])*np.maximum(abs(self.Z1[0])-self.lamd1*self.W/self.mu, 0)
            self.Z1[1] = np.sign(self.Z1[1])*np.maximum(abs(self.Z1[1])-self.lamd1*self.W/self.mu, 0)
        else:
            self.Z1 = updateZ.multiZ(self.A, self.mu, self.lamd1*self.W, self.lamd2*self.W) #PRIOR
        '''
        for i in range(self.p):
            for j in range(self.p):
                self.Z1[0][i][j],self.Z1[1][i][j] = tuple(soft_threshold(x,self.lamd1/self.mu) for x in self.fused_lasso(i,j))
        '''


    def update_us(self):
        #self.U0 = copy.deepcopy(self.U1)
        for i in range(self.k):
            self.U1[i] = self.U1[i]+self.THETA1[i]-self.Z1[i]

    def fused_func(self):
        # the less the better
        vfunc = 0
        for i in range(self.k):
            vfunc -= self.nk[i] * (np.log(LA.det(self.THETA1[i]))-np.trace(np.dot(self.S[i],self.THETA1[i])))
            #vfunc = vfunc + self.rho/2 * (np.linalg.norm(self.THETA1[i]-self.Z1[i]+self.U1[i],'fro') - np.linalg.norm(self.U1[i]))
            tmp  = self.THETA1[i] - np.diag(np.diag(self.THETA1[i]))
            vfunc += self.lamd1 * LA.norm(tmp.reshape(self.p*self.p),1)
        secTerm = self.THETA1[0]-self.THETA1[1]
        vfunc += self.lamd2 * LA.norm(secTerm.reshape(self.p*self.p),1)
        return vfunc

    def update_Thetas_JGL(self):
        '''updateing Thetas on full matrix without pathways'''
        self.THETA0 = copy.copy(self.THETA1)
        for i in range(self.k):
            self.THETA1[i] = self.update_Theta_JGL(self.S[i], self.Z1[i], self.U1[i], self.mu, self.nk[i])

    def update_Theta_JGL(self,S,Z,U,mu,nk):
        '''called by update_Thetas_JGL()'''
        D,vec = LA.eig(S + mu/nk*(U-Z))
        Dhat = nk/2/mu*np.diag(-D + np.sqrt(np.power(D,2)+4*mu/nk))
        return np.dot(vec, np.dot(Dhat,vec.T))

    def admm_for_mjgl(self):
        self.initializing()
        notconverged = True
        funcValues = []
        ITER = 0
        while notconverged:
            #print 'ITER =',ITER,'======================================================================================' 
            ITER = ITER+1
            if ITER%20==0:
                print 'ITER=', ITER
            self.update_thetas()
            #self.update_Thetas_JGL()
            self.update_zs()
            self.update_us()
            newMu = self.mu
            self.mu = newMu if newMu < 1e5 else 1e5
            numerator = sum(map(lambda x,y: np.linalg.norm(x-y,"fro"),self.THETA1,self.THETA0))
            denominator = sum(np.linalg.norm(x,"fro") for x in self.THETA0)
            ratio = numerator / denominator
            #funcValue = self.fused_func()
            #funcValues.append(funcValue)
            #print "ratio = ",ratio
            #print "F-NORM THETA1", LA.norm(self.THETA1[0],'fro')
            #print "F-NORM THETA0", LA.norm(self.THETA0[0],'fro')
            #print "objective function value = ",funcValue
            if numerator / denominator < self.tol:
                notconverged = False
            elif ITER > self.maxiter:
                print "ADMM MAX ITERATION REACHED"
                notconverged = False
        print 'ITER=', ITER
        return None 

