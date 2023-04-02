import numpy as np
import matplotlib.pyplot as plt
#import g3_linear_solver as g3
import pulp as pl
from multiprocessing import Process

d = 4

def v(x, j, n):
    if n == 1:
        return [1, (3 - (4/(d-2))*(j*(j+d-3)))/((1+x)), 0.5/((1+x))**2, ((j*(j+d-3))*(2*(j*(j+d-3))-5*d+4))/((1+x))**2]
    
    elif n == 2:
        return [1, (3 - (4/(d-2))*(j*(j+d-3)))/((1+x)), 0.5/((1+x))**2,((j*(j+d-3))*(2*(j*(j+d-3))-5*d+4))/((1+x))**2, \
            ((23*d**2-12*d-20)*j*(j+d-3) + (-21*d-2)*(j*(j+d-3))**2 + 4*(j*(j+d-3))**3)/((1+x))**3]
    
    elif n ==3:
        return [1, (3 - (4/(d-2))*(j*(j+d-3)))/((1+x)), 0.5/((1+x))**2, ((j*(j+d-3))*(2*(j*(j+d-3))-5*d+4))/((1+x))**2, \
            ((23*d**2-12*d-20)*j*(j+d-3) + (-21*d-2)*(j*(j+d-3))**2 + 4*(j*(j+d-3))**3)/((1+x))**3, \
            (4*(13*d**2+21*d+2)*(j*(j+d-3))**2 -3*(d+2)*(17*d**2+4*d-16)*j*(j+d-3) + 2*(-9*d-8)*(j*(j+d-3))**3 + 2*(j*(j+d-3))**4)/((1+x))**4]
    elif n == 5:
        return [1, (3 - (4/(d-2))*(j*(j+d-3)))/((1+x)), 0.5/((1+x))**2, ((j*(j+d-3))*(2*(j*(j+d-3))-5*d+4))/((1+x))**2, \
            ((23*d**2-12*d-20)*j*(j+d-3) + (-21*d-2)*(j*(j+d-3))**2 + 4*(j*(j+d-3))**3)/((1+x))**3, \
            (4*(13*d**2+21*d+2)*(j*(j+d-3))**2 -3*(d+2)*(17*d**2+4*d-16)*j*(j+d-3) + 2*(-9*d-8)*(j*(j+d-3))**3 + 2*(j*(j+d-3))**4)/((1+x))**4, \
            (6*(45*d**2+140*d+92)*(j*(j+d-3))**3 + 4*(-140*d**3-619*d**2-711*d-46)*(j*(j+d-3))**2 + (401*d**4 +2108*d**3 + 2284*d**2 - 3008*d -3360)*(j*(j+d-3)) + 5*(-11*d -18)*(j*(j+d-3))**4 + 4*(j*(j+d-3))**5)/((1+x))**5, \
            ((-27*d-14)*(j*(j+d-3))**2 + 2*(d-1)*(19*d+22)*j*(j+d-3) + 4*(j*(j+d-3))**3)/((1+x))**6]

def g4_min(g3_A, n, J, X, print_sol = False):

    solver = pl.GLPK_CMD(msg=0)

    prob = pl.LpProblem("g4", pl.LpMaximize)

    A = pl.LpVariable("A")

    B = pl.LpVariable("B")

    C = pl.LpVariable("C")

    if n >= 2:
        D = pl.LpVariable("D")
    if n>=3:
        E = pl.LpVariable("E")
    if n>=5:
        F = pl.LpVariable("F")
        G = pl.LpVariable("G")

    prob += A + B*g3_A # objective function


    if n == 1:
        y = [-A, -B, 1, C]
        prob += np.dot(y, [0, 0, 0, 2]) >= 0
    elif n == 2:
        y = [-A, -B, 1, C, D]
        prob += np.dot(y, [0, 0, 0, 0, 2]) >= 0
    elif n == 3:
        y = [-A, -B, 1, C, D, E]
        prob += np.dot(y, [0, 0, 0, 0, 0, 2]) >= 0
    elif n == 5:
        y = [-A, -B, 1, C, D, E, F, G]
        prob += np.dot(y, [0, 0,0,  0, 0, 0, 2, 2]) >= 0
    for j in J:
        for x in X:
            f = np.dot(v(x,j, n), y)
            prob += f >= 0 # constraint

    #prob += np.dot(y, [0, 0, 0, 0, 2]) >= 0
    prob.writeLP("g4.lp")
    prob.solve(solver)

    vals = []
    for var in prob.variables():
        if print_sol:
            print(var.name, "=", var.varValue)
        vals.append(var.varValue)

    vals = np.array(vals)
    vals = np.insert(vals, 2, 1)
    return vals

def g4_max(g3_A, n, J, X, print_sol = False):

    solver = pl.GLPK_CMD(msg=0)

    prob = pl.LpProblem("g4", pl.LpMinimize)

    A = pl.LpVariable("A")

    B = pl.LpVariable("B")

    C = pl.LpVariable("C")

    if n >= 2:
        D = pl.LpVariable("D")
    if n>=3:
        E = pl.LpVariable("E")
    if n>=5:
        F = pl.LpVariable("F")
        G = pl.LpVariable("G")

    prob += A + B*g3_A # objective function


    if n == 1:
        y = [A, B, -1, C]
        prob += np.dot(y, [0, 0, 0, 2]) >= 0
    elif n == 2:
        y = [A, B, -1, C, D]
        prob += np.dot(y, [0, 0, 0, 0, 2]) >= 0
    elif n == 3:
        y = [A, B, -1, C, D, E]
        prob += np.dot(y, [0, 0, 0, 0, 0, 2]) >= 0
    elif n == 5:
        y = [A, B, -1, C, D, E, F, G]
        prob += np.dot(y, [0, 0,0,  0, 0, 0, 2, 2]) >= 0
    for j in J:
        for x in X:
            f = np.dot(v(x,j, n), y)
            prob += f >= 0 # constraint

    #prob += np.dot(y, [0, 0, 0, 0, 2]) >= 0
    prob.writeLP("g4.lp")
    prob.solve(solver)

    vals = []
    for var in prob.variables():
        if print_sol:
            print(var.name, "=", var.varValue)
        vals.append(var.varValue)

    vals = np.array(vals)
    vals = np.insert(vals, 2, 1)
    return vals


n = int(input("Enter the Number of constraints (1, 2, 3, 5): "))
g3_A = -10
Js = np.arange(0, 40, 2)
Xs = np.linspace(0, 10, 1000, endpoint=True)
vals = g4_min(g3_A, n, Js, Xs, print_sol=True)
print("g4~ >= ", vals[0] + g3_A*vals[1], "for g3 = ", g3_A)
print("\n")
vals = g4_max(g3_A, n, Js, Xs, print_sol=True)
print(vals)
print("g4~ <= ", vals[0] + g3_A*vals[1], "for g3 = ", g3_A)
print("\n")

def plot_loops(i, g3_s, n, Js, Xn, Mat, dx, dy, pad):
    sol1 = g4_min(g3_s[i], n, Js, Xn)
    h1 = sol1[0] + g3_s[i]*sol1[1]
    sol2 = g4_max(g3_s[i], n, Js, Xn)
    h2 = sol2[0] + g3_s[i]*sol2[1]
    print(i,g3_s[i], h1, h2)
    if h2-h1 > 0:
        h1 = int(np.round(h1, 2)*int(1/dy))
        h2 = int(np.round(h2, 2)*int(1/dy))
        Mat[i+pad, h1:h2] = 1
    
def plot():
    print("\nPlotting the g3~ g4~ plane")
    dx = 0.1
    dy = 0.01
    pad = 5
    Mat = np.zeros((int(14/dx)+2*pad, int(0.5/dy)+2*pad))
    print(Mat.shape)
    Xn = np.linspace(0, 2, 200, endpoint=True)
    g3_s = np.linspace(-11, 3, int(14/dx), endpoint=True)
    x_tick = np.round(np.linspace(-11-pad*dx, 3+pad*dx, 11, endpoint=True), 2)
    y_tick = np.round(np.linspace(0-pad*dy, 0.5+pad*dy, 11, endpoint=True), 2)
    print(g3_s.shape)
    processes = [Process(target=plot_loops, args=(i, g3_s, n, Js, Xn, Mat, dx, dy, pad)) for i in range(int(14/dx))]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    Mat[:, -pad:] = 0

    plt.figure(figsize=(10, 10))
    plt.imshow(Mat.transpose(), cmap='PuBu', aspect='auto', origin='lower', alpha=0.5)
    plt.xlabel("g3~")
    plt.ylabel("g4~")
    plt.title("g3~ g4~ plane")
    plt.xticks(np.linspace(0, int(14/dx)+2*pad, 11, endpoint=True), x_tick)
    plt.yticks(np.linspace(0, int(0.5/dy)+2*pad, 11, endpoint=True), y_tick)
    plt.minorticks_on()
    plt.grid()
    plt.show()

plot()