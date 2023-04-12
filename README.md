# Extremal Effective Field Theories
## A project by Aryaman Bhutani done in partial fulfillment of the requirements for the course PH354: Computational Physics at IISc Bangalore

### Course Instructor: Prof. Manish Jain, Department of Physics, Indian Institute of Science
### Project Mentor: Prof. Aninda Sinha, Centre for High Energy Physics, Indian Institute of Science

This project aims to reproduce the results of Caron-Huot, S., Van Duong, V. Extremal effective field theories. J. High Energ. Phys. 2021, 280 (2021). [doi](https://doi.org/10.1007/JHEP05(2021)280) in partial. In particular, the optimal bounds for g3 upto Mandelstam order 7 and g3-g4 exclusion plots have been obtained. The techniques used involve a brute force optimisation method as a naive solution attempt along with Linear Programming. Linear programming was achieved by using a python wrapper package (PuLP) for various linear solvers (GLPK, MOSEK, Gurobi). For details regarding the project and the implementation techniques, please refer to the [Report]. 

If you wish to reproduce the results using these codes, please ensure the following:
Recommended OS: Linux (Ubuntu 20.04 or greater)
Windows can be used, however, installation of GLPK on Windows has not been tested. MacOS has not been tested.
PuLP can be installed by PIP `python -m pip install pulp` or from conda by `conda install -c conda-forge pulp`
The codes can make use of GLPK/MOSEK/Gurobi. Other linear solvers which are compatible with PuLP can also be utilised. It is recommended to use GLPK/MOSEK as they have been tested for consistent results. 
GLPK can be installed on LINUX using apt install by `$ sudo apt install glpk`. Several other wrappers for GLPK are also available for python, notably [scikit-glpk](https://pypi.org/project/scikit-glpk/) and [PyGLPK](https://pypi.org/project/glpk/). To install GLPK for other OS, follow the links for [Windows](https://winglpk.sourceforge.net/) or on Mac by `brew install glpk`. 

## File Descriptions

[3_sum_rule.py](https://github.com/Ary276/Extremal_Effective_Field_Theories_PH354_Project/blob/master/3_sum_rule.py) is an implementation to solve the Three Sum Rule Warm Up problem presented in [ref](https://arxiv.org/abs/2011.02957v2).

[brute_force.py](https://github.com/Ary276/Extremal_Effective_Field_Theories_PH354_Project/blob/master/brute_force.py) is an implementation to find the lower bound for g3 based on a brute force approach. This brute force search strategy is optimised for better and faster convergence. For more details about its implementation, check [report]. Inputs required: Mandelstam order upto which to compute bound.

[g3_linear_solver.py](https://github.com/Ary276/Extremal_Effective_Field_Theories_PH354_Project/blob/master/g3_linear_solver.py) is an implementation whcih uses Linear Programming to obtain the lower bound for the parameter g3. Input required: Number of constraint equations to use.

[g4_linear_solver.py](https://github.com/Ary276/Extremal_Effective_Field_Theories_PH354_Project/blob/master/g4_linear_solver.py) is an implementation whcih uses Linear Programming to obtain the lower bound and upper bound for the parameter g4 for a given value of g3. It also plots the feasible region in the g3-g4 plane. Input required: Number of constraint equations to use and test value of g3.

Some of these codes can take a long time to run. Particularly brute_force.py with input as 6 or 7 and g4_linear_solver.py if plotting is done. 

### Package requirements
(requirements are not strict, but recommendations)
`python >= 3.10.8`
`numpy >= 1.23.5`
`matplotlib >= 3.7.1`
`pulp >= 2.7.0`
`multiprocessing`
`time`

By Aryaman Bhutani
Bachelor of Science (Research)
Indian Institute of Science
