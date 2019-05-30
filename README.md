# WFPGL

An implements of Weighted Fused Pathway Graphical Lasso that should run in python2.7+ and Linux.
This implement requires the package of the modified version of Pathway Graphical Lasso (PGL), which is built based on PGL v1.0.2 [https://sites.google.com/a/cs.washington.edu/pathglasso/](https://sites.google.com/a/cs.washington.edu/pathglasso/) and setup firsly.

#installation
To install PGL, run:
```python
python setup.py build_ext --inplace
```
Dependencies: numpy,cython,networkx

#Data Decription
We provide preprocessed data of diabetets and breast cancer in each directories, and as well as a synthetic dataset for simulation.

#Run the algorithm
The example.py shows how to perform WFPGL on the synthetic data.
