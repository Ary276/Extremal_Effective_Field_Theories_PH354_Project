import numpy as np
import matplotlib.pyplot as plt

J = np.arange(0, 20, 2)
X = np.linspace(0, 2, 100, endpoint=True)


d = 4

def v(x, j):
    return [1, (3 - (4/(d-2))*(j*(j+d-3)))/((1+x)), ((j*(j+d-3))*(2*(j*(j+d-3))-5*d+4))/((1+x))**2]
            #((23*d**2-12*d-20)*j*(j+d-3) + (-21*d-2)*(j*(j+d-3))**2 + 4*(j*(j+d-3))**3)/((1+x))**3]


a = 0
b = 0
g = 0
n = 20
ans = 10.6125
eps = 1e-5
k = 0
while(k<20 and abs(a-ans)>eps):
    alpha = np.random.normal(a, 2**(4-k), n)
    #print(alpha)
    beta = np.random.normal(b, 2**(-2-k), n)
    gamma = np.random.normal(g, 2**(-6-k), n)

    coeff_list = np.array([[], []])

    for A in alpha:
        for B in beta:
            Flag = False
            for j in J:
                for x in X:
                    y = np.array([A, 1, B])
                    f = np.dot(v(x,j), y)
                    if f < 0:
                            Flag = True
                            break
                    else:
                        continue
                
                if Flag == True:
                    break
                else:
                    continue

            if Flag == False:
                coeff_list = np.hstack((coeff_list, np.array([[A], [B]])))
            else:
                continue

    if coeff_list.size == 0:
        break
    else:
        print(np.min(coeff_list[0, :]))
        a = np.min(coeff_list[0, :])
        ind = np.argmin(coeff_list[0, :])
        b = coeff_list[1, ind]
    k+=1

print('k = ', k)
print('alpha = ', a)
print('beta = ', b)

def f(x, j):
    return a*v(x,j)[0]/(1+x)**2 + v(x,j)[1]/(1+x)**3 + b*v(x,j)[2]/(1+x)**4

plt.figure(figsize=(10, 8))
plt.plot(1+X, f(X, 0), label = 'J = 0')
plt.plot(1+X, f(X, 2), label = 'J = 2')
plt.plot(1+X, f(X, 4), label = 'J = 4')
plt.plot(1+X, f(X, 6), label = 'J = 6')
plt.plot(1+X, f(X, 8), label = 'J = 8')
plt.ylim(0, 5)
plt.legend()
plt.show()