import numpy as np
import matplotlib.pyplot as plt

# matrix multiplication symbol requires Python 3.x. You can convert it to a version that would run on 2.x.

def IRLS(A, y, p, epsilon, x0):
    MaxIter = 10000
    
    x = x0
    w = (x**2 + epsilon)**(p / 2 - 1)
    
    for i in range(MaxIter):
        x_old = x.copy()
        Q = np.diag(1 / np.sqrt(w).flatten())

        [Q0, R] = np.linalg.qr((A @ Q).T)
        x = Q @ Q0 @ np.linalg.solve(R.T, y)
        w = (x**2 + epsilon)**(p/2 - 1)
        if np.linalg.norm(x - x_old) < 1e-6:
            print('Converged.')
            break
    return x

M = 30
N = 100
K = 5
A = np.random.normal(size=(M, N)) / np.sqrt(M)
s =1 * (np.random.uniform(size=(N,1)) < K/N)
y = A @ s
epsilon = 1e-5
p = 1
x0 = np.random.normal(size=(N,1))
x = IRLS(A, y, p, epsilon, x0)
res = []
for ind, i in enumerate(s):
    if i == 1:
        res.append(ind)
print(res)
print(np.round(x) - s == np.zeros(100))
plt.figure()
plt.plot(x)
plt.plot(s, linestyle='--')
plt.show()

