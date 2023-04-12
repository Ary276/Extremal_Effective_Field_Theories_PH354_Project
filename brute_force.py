import numpy as np
import matplotlib.pyplot as plt
import itertools
import multiprocessing as mp
import time

d = 4
def v_i(j_x, order):
    x = j_x[1]
    j = j_x[0]
    if order == 3:
        return np.asarray([1, (3 - (4/(d-2))*(j*(j+d-3)))/((1+x)), ((j*(j+d-3))*(2*(j*(j+d-3))-5*d+4))/((1+x))**2], dtype=np.float64)

    elif order == 4:
        return np.asarray([1, (3 - (4/(d-2))*(j*(j+d-3)))/((1+x)), ((j*(j+d-3))*(2*(j*(j+d-3))-5*d+4))/((1+x))**2, \
            ((23*d**2-12*d-20)*j*(j+d-3) + (-21*d-2)*(j*(j+d-3))**2 + 4*(j*(j+d-3))**3)/((1+x))**3], dtype=np.float64)

    elif order == 5:
        return [1, (3 - (4/(d-2))*(j*(j+d-3)))/((1+x)), ((j*(j+d-3))*(2*(j*(j+d-3))-5*d+4))/((1+x))**2, \
            ((23*d**2-12*d-20)*j*(j+d-3) + (-21*d-2)*(j*(j+d-3))**2 + 4*(j*(j+d-3))**3)/((1+x))**3, \
            (4*(13*d**2+21*d+2)*(j*(j+d-3))**2 -3*(d+2)*(17*d**2+4*d-16)*j*(j+d-3) + 2*(-9*d-8)*(j*(j+d-3))**3 + 2*(j*(j+d-3))**4)/((1+x))**4]
    
    elif order == 7:
        return [1, (3 - (4/(d-2))*(j*(j+d-3)))/((1+x)), ((j*(j+d-3))*(2*(j*(j+d-3))-5*d+4))/((1+x))**2, \
            ((23*d**2-12*d-20)*j*(j+d-3) + (-21*d-2)*(j*(j+d-3))**2 + 4*(j*(j+d-3))**3)/((1+x))**3, \
            (4*(13*d**2+21*d+2)*(j*(j+d-3))**2 -3*(d+2)*(17*d**2+4*d-16)*j*(j+d-3) + 2*(-9*d-8)*(j*(j+d-3))**3 + 2*(j*(j+d-3))**4)/((1+x))**4, \
            (6*(45*d**2+140*d+92)*(j*(j+d-3))**3 + 4*(-140*d**3-619*d**2-711*d-46)*(j*(j+d-3))**2 + (401*d**4 +2108*d**3 + 2284*d**2 - 3008*d -3360)*(j*(j+d-3)) + 5*(-11*d -18)*(j*(j+d-3))**4 + 4*(j*(j+d-3))**5)/((1+x))**5, \
            ((-27*d-14)*(j*(j+d-3))**2 + 2*(d-1)*(19*d+22)*j*(j+d-3) + 4*(j*(j+d-3))**3)/((1+x))**6]
    else:
        return np.asarray([1.], dtype=np.float64)

def coeff(order, vals, k, n):
    if order == 3:
        return np.array([np.random.normal(vals[0], 2**(4-k), n), np.full(n, 1.), np.random.normal(vals[2], 2**(-2-k), n)], dtype=np.float64)
    elif order == 4:
        return np.array([np.random.normal(vals[0], 2**(4-k), n), np.full(n, 1.), np.random.normal(vals[2], 2**(-2-k), n), np.random.normal(vals[3], 2**(-4-k), n)], dtype=np.float64)
    elif order == 5:
        return np.array([np.random.normal(vals[0], 2**(4-k), n), np.full(n, 1.), np.random.normal(vals[2], 2**(-2-k), n), np.random.normal(vals[3], 2**(-4-k), n), np.random.normal(vals[4], 2**(-6-k), n)], dtype=np.float64)
    elif order == 7:
        return np.array([np.random.normal(vals[0], 2**(4-k), n), np.full(n, 1.), np.random.normal(vals[2], 2**(-2-k), n), np.random.normal(vals[3], 2**(-4-k), n), np.random.normal(vals[4], 2**(-6-k), n), np.random.normal(vals[5], 2**(-8-k), n), np.random.normal(vals[6], 2**(-10-k), n)], dtype=np.float64)
    else:
        return np.array([np.random.normal(vals[0], 2**(4-k), n)], dtype=np.float64)

def loop(J_X, coeff0, it, order):
    coeff_list = np.zeros([0, order], dtype=np.float64)
    A = coeff0
    for y in it:
        Flag = False
        y = np.insert(y, 0, [A, 1])
        for j_x in J_X:
            f = np.dot(v_i(j_x, order), y)
            if f < 0:
                Flag = True
                break
            else:
                continue
                
        if Flag == False:
            coeff_list = np.vstack((coeff_list, y))
        else:
            continue
    return coeff_list

def brute_force(J_X, vals, order, n):
    vals_old = np.zeros_like(vals)
    eps = 1e-5
    k = 0
    while(k<20 and np.linalg.norm(vals-vals_old)>eps):
        coeffs = coeff(order,vals, k, n)
        it = list(itertools.product(*coeffs[2:]))
        #print(coeffs[0])
        #print(it[0])
        p = mp.Pool(mp.cpu_count())
        coeff_list = np.array(p.starmap(loop, [(J_X, coeffs[0, i], it, order) for i in range(n)]), dtype=object)
        p.close()
        coeff_list = np.vstack(coeff_list)
        if coeff_list.size == 0:
            break
        else:
            print("Current Guess for A: ", np.min(coeff_list[:, 0]))
            #print(coeff_list)
            ind = np.argmin(coeff_list[:, 0])
            vals_old = vals
            vals = coeff_list[ind, :]
        k+=1
    return vals, k


if __name__ == '__main__':
    print("Note: Order 6 and 7 will take really long to run")
    order = int(input('Enter the order to be computed (4/5/6/7): '))
    order -= 1
    start = time.time()
    J = np.arange(0, 40, 2)
    X = np.linspace(0, 5, 100, endpoint=True)
    J_X = np.array(list(itertools.product(J, X)))
    if order == 3:
        vals = np.array([0, 1, 0])
    elif order == 4:
        vals = np.array([0, 1, 0, 0])
    elif order == 5:
        vals = np.array([0, 1, 0, 0, 0])
    elif order == 7:
        vals = np.array([0, 1, 0, 0, 0, 0, 0])
    out, k = brute_force(J_X, vals, order, 20)
    end = time.time()
    print(out)
    print('k = ', k)
    print('A = ', out[0])
    print('B = ', out[2])
    print("Hence, the solution is: A>=", -out[0])
    print('Time: ', end-start)

#def f(x, j):
#    return a*v(x,j)[0]/(1+x)**2 + v(x,j)[1]/(1+x)**3 + b*v(x,j)[2]/(1+x)**4

#plt.figure()
#plt.plot(1+X, f(X, 0), label = 'J = 0')
#plt.plot(1+X, f(X, 2), label = 'J = 2')
#plt.plot(1+X, f(X, 4), label = 'J = 4')
#plt.plot(1+X, f(X, 6), label = 'J = 6')
#plt.plot(1+X, f(X, 8), label = 'J = 8')
#plt.ylim(0, 2.5)
#plt.legend()
#plt.show()