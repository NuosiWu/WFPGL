#coding=utf-8
import numpy as np
import pandas as pd
import pglasso
from sklearn.preprocessing import scale
import WFPGL


def load_pathways():
	p0 = open("pathways.csv").readlines()
	p1 = map(lambda x: x.split(":")[1].split(","), p0)
	p2 = map(lambda x: map(str.strip, x), p1)
	return p2

def map_pathways(gene_names, p2):
	p3 = map(lambda x: map(gene_names.index, x), p2)
	return p3

# read data and prior information
X1df = pd.read_csv("X_1.csv",header=None)
X2df = pd.read_csv("X_2.csv",header=None)
prior = np.array(pd.read_csv("net_prior.csv",header=None)) # prior regulatory network
print prior.shape

pathways = load_pathways()
all_genes = []
gene_names = map(str,X1df.index)
map(all_genes.extend, pathways)
all_genes = sorted(list(set(all_genes)),key=lambda x: gene_names.index(x))
pathways = map_pathways(all_genes, pathways) # prior pathways 

idxes = np.array([gene_names.index(g) for g in all_genes])
X1 = scale(np.array(X1df)[idxes,:].T,with_std=False).T
X2 = scale(np.array(X2df)[idxes,:].T,with_std=False).T
Y = [np.array(X1),np.array(X2)]


# run the algorithm
lambda1 = .05
lambda2 = .01
ONEsetting = WFPGL.WFPGL(Y,pathways,lambda1,lambda2,prior,"fused",mu=1,maxiter=1000)
funcValue = ONEsetting.admm_for_mjgl() # value of objective function
output = ONERUN.THETA1 # output precision matrices



