import numpy as np
import matplotlib.pyplot as plt
#import g3_linear_solver as g3
import pulp as pl
import multiprocessing as mp
import time
#import mosek
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

    solver = pl.GLPK_CMD(msg=0, mip=False)
    #solver = pl.MOSEK(msg=0, mip=False, sol_type=mosek.soltype.itr)
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
    if print_sol:
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

    solver = pl.GLPK_CMD(msg=0, mip=False)
    #solver = pl.MOSEK(msg=0, mip=False, sol_type=mosek.soltype.itr)
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
    if print_sol:
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


def plot_loop(i, g3_s, n, Js, Xn, dy):
    sol1 = g4_min(g3_s[i], n, Js, Xn)
    h1 = sol1[0] + g3_s[i]*sol1[1]
    sol2 = g4_max(g3_s[i], n, Js, Xn)
    h2 = sol2[0] + g3_s[i]*sol2[1]
    #print(i,g3_s[i], h1, h2)
    if h2-h1 > 0:
        h1 = np.rint(h1/dy).astype(int)
        h2 = np.rint(h2/dy).astype(int)
        return np.array([i, h1, h2])
    else:
        return np.array([i, 0, 0])
    
def plot():
    print("\nPlotting the g3~ g4~ plane")
    dx = 0.01
    dy = 0.00001
    Xn = np.linspace(0, 100, 500, endpoint=True)
    g3_s = np.linspace(-11, 3, int(14/dx), endpoint=True)
    x_tick = np.round(np.linspace(-11, 3, 11, endpoint=True), 2)
    y_tick = np.round(np.linspace(0, 0.5, 11, endpoint=True), 2)
    print(g3_s.shape)
    #for i in range(int(14/dx)):
    #    plot_loop(i, g3_s, n, Js, Xn, Mat, dx, dy, pad)
    pool = mp.Pool(mp.cpu_count())
    results = np.array(pool.starmap(plot_loop, [(i, g3_s, n, Js, Xn, dy) for i in range(int(14/dx))]))
    pool.close()
    print(results)
    Mat = np.zeros((int(14/dx), int(0.5/dy)))
    for i in range(results.shape[0]):
        Mat[results[i, 0], results[i, 1]:results[i, 2]] = 1
    print(Mat.shape)
    mx, my = Mat.shape
    print(results.shape)
    pad = 2000
    padx = int(1/dx)
    plt.figure(figsize=(10, 10))
    plt.imshow(Mat.transpose(), cmap='RdPu', aspect='auto', origin='lower', alpha=0.5)
    plt.xlim(0-padx, mx+padx)
    plt.ylim(0-pad, my+pad)
    plt.xlabel("g3~")
    plt.ylabel("g4~")
    plt.title("g3~ g4~ plane for n = "+str(n+3))
    plt.xticks(np.linspace(0, int(14/dx), 11, endpoint=True), x_tick)
    plt.yticks(np.linspace(0, int(0.5/dy), 11, endpoint=True), y_tick)
    plt.savefig("g3_g4_plane_n="+str(n+3)+"ayaya.png", dpi=1000)
    plt.show()


if __name__ == "__main__":
    n = int(input("Enter the Number of constraints (1, 2, 3, 5): "))
    start = time.time()
    g3_A = float(input("Enter the value of g3~ (between -11 and 3): "))
    Js = np.arange(0, 40, 2)
    Xs = np.linspace(0, 100, 500, endpoint=True)
    print("\nMinimum")
    vals = g4_min(g3_A, n, Js, Xs, print_sol=True)
    print("g4~ >= ", vals[0] + g3_A*vals[1], "for g3 = ", g3_A)
    print("\nMaximum")
    vals = g4_max(g3_A, n, Js, Xs, print_sol=True)
    print(vals)
    print("g4~ <= ", vals[0] + g3_A*vals[1], "for g3 = ", g3_A)
    print("\n")
    #plot()
    end = time.time()
    print("\nTime taken: ", end-start)