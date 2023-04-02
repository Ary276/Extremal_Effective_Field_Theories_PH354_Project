# Testing out linear solver

import numpy as np
import matplotlib.pyplot as plt
import pulp as pl


solver = pl.GLPK_CMD()
#solver = pl.GUROBI_CMD()

prob = pl.LpProblem("g3", pl.LpMaximize)

n = int(input("Enter the Number of constraints (1, 2, 3, 5): "))
J = np.arange(0, 40, 2)
X = np.linspace(0, 10, 1000, endpoint=True)


d = 4

def v(x, j, n):
    if n == 1:
        return [1, (3 - (4/(d-2))*(j*(j+d-3)))/((1+x)), ((j*(j+d-3))*(2*(j*(j+d-3))-5*d+4))/((1+x))**2]
    
    elif n == 2:
        return [1, (3 - (4/(d-2))*(j*(j+d-3)))/((1+x)), ((j*(j+d-3))*(2*(j*(j+d-3))-5*d+4))/((1+x))**2, \
            ((23*d**2-12*d-20)*j*(j+d-3) + (-21*d-2)*(j*(j+d-3))**2 + 4*(j*(j+d-3))**3)/((1+x))**3]
    
    elif n ==3:
        return [1, (3 - (4/(d-2))*(j*(j+d-3)))/((1+x)), ((j*(j+d-3))*(2*(j*(j+d-3))-5*d+4))/((1+x))**2, \
            ((23*d**2-12*d-20)*j*(j+d-3) + (-21*d-2)*(j*(j+d-3))**2 + 4*(j*(j+d-3))**3)/((1+x))**3, \
            (4*(13*d**2+21*d+2)*(j*(j+d-3))**2 -3*(d+2)*(17*d**2+4*d-16)*j*(j+d-3) + 2*(-9*d-8)*(j*(j+d-3))**3 + 2*(j*(j+d-3))**4)/((1+x))**4]
    elif n == 5:
        return [1, (3 - (4/(d-2))*(j*(j+d-3)))/((1+x)), ((j*(j+d-3))*(2*(j*(j+d-3))-5*d+4))/((1+x))**2, \
            ((23*d**2-12*d-20)*j*(j+d-3) + (-21*d-2)*(j*(j+d-3))**2 + 4*(j*(j+d-3))**3)/((1+x))**3, \
            (4*(13*d**2+21*d+2)*(j*(j+d-3))**2 -3*(d+2)*(17*d**2+4*d-16)*j*(j+d-3) + 2*(-9*d-8)*(j*(j+d-3))**3 + 2*(j*(j+d-3))**4)/((1+x))**4, \
            (6*(45*d**2+140*d+92)*(j*(j+d-3))**3 + 4*(-140*d**3-619*d**2-711*d-46)*(j*(j+d-3))**2 + (401*d**4 +2108*d**3 + 2284*d**2 - 3008*d -3360)*(j*(j+d-3)) + 5*(-11*d -18)*(j*(j+d-3))**4 + 4*(j*(j+d-3))**5)/((1+x))**5, \
            ((-27*d-14)*(j*(j+d-3))**2 + 2*(d-1)*(19*d+22)*j*(j+d-3) + 4*(j*(j+d-3))**3)/((1+x))**6]

A = pl.LpVariable("A")

B = pl.LpVariable("B")

if n >= 2:
    C = pl.LpVariable("C")
if n>=3:
    D = pl.LpVariable("D")
if n>=5:
    E = pl.LpVariable("E")
    F = pl.LpVariable("F")

prob += A # objective function


if n == 1:
    y = [-A, 1, B]
    prob += np.dot(y, [0, 0, 2]) >= 0
elif n == 2:
    y = [-A, 1, B, C]
    prob += np.dot(y, [0, 0, 0, 2]) >= 0
elif n == 3:
    y = [-A, 1, B, C, D]
    prob += np.dot(y, [0, 0, 0, 0, 2]) >= 0
elif n == 5:
    y = [-A, 1, B, C, D, E, F]
    prob += np.dot(y, [0, 0, 0, 0, 0, 2, 2]) >= 0
for j in J:
    for x in X:
        f = np.dot(v(x,j, n), y)
        prob += f >= 0 # constraint

#prob += np.dot(y, [0, 0, 0, 0, 2]) >= 0
prob.writeLP("g3.lp")
prob.solve(solver)

vals = []
for var in prob.variables():
    print(var.name, "=", var.varValue)
    vals.append(var.varValue)
vals = np.array(vals)
vals = np.insert(vals, 1, 1)
vals[0] = -vals[0]
print("g3~ >=", vals[0])
X2 = np.linspace(0, 2, 1000)
f1 = np.zeros(len(X2))
f2 = np.zeros(len(X2))
f3 = np.zeros(len(X2))
f4 = np.zeros(len(X2))

for i in range(len(vals)):
    f1 += v(X2, 0, n)[i]*vals[i]/(1+X2)**(i+2)
    f2 += v(X2, 2, n)[i]*vals[i]/(1+X2)**(i+2)
    f3 += v(X2, 4, n)[i]*vals[i]/(1+X2)**(i+2)
    f4 += v(X2, 6, n)[i]*vals[i]/(1+X2)**(i+2)


plt.figure(figsize=(10,10))
plt.title("Optimal Solution for n = {}".format(n+3))
plt.plot(X2, f1, label="j=0")
plt.plot(X2, f2, label="j=2")
plt.plot(X2, f3, label="j=4")
plt.plot(X2, f4, label="j=6")
plt.ylim(0, 5)
plt.legend()
plt.show()
