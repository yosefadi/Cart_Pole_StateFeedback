import numpy as np

l = 0.5
mp = 0.1
mc = 1.0
g = 9.8
dt = 0.02  # from openai gym docs

Ar = np.array([[0, 0, 1, 0],
               [0, 0, 0, 1],
               [0, -mp*(mp * (g-l) + mc*g)/((mc+mp)*((4/3) * mc + (1/3) * mp)), 0, 0],
               [0, (mp*(g-l) + mc * g)/(l*((4/3) * mc + (1/3) * mp)), 0, 0]])

print("Ar: \n", Ar)

Aaa = Ar[:2,:2]
print("Aaa: \n", Aaa)

Aau = Ar[:2,2:]
print("Aau: \n", Aau)

Aua = Ar[2:,:2]
print("Aua: \n", Aua)

Auu = Ar[2:,2:]
print("Auu: \n", Auu)

Br = np.array([[0],
              [0],
              [(1/(mc + mp) - mp/((mc + mp) * ((4/3) * mc + (1/3) * mp)))],
              [(-1/(l * ((4/3) * mc + (1/3) * mp)))],])
print("Br: \n", Br)

Ba = Br[:2]
Bb = Br[2:]
print("Ba: \n", Ba)
print("Bb: \n", Bb)

x = np.array([99, 88, 77, 66])
x[[2,1]] = x[[1,2]]
print("x: \n", x)

A = np.empty([4,4])
A[[0, 1, 2, 3]] = Ar[[0, 2, 1, 3]]
A[:, [1, 2]] = A[:, [2, 1]]
print("A: \n", A)

B = np.empty([4,1])
B[[0, 1, 2, 3]] = Br[[0, 2, 1, 3]]
print("B: \n", B)

x_hat_new = np.empty([4,])
print("x_hat_new: \n", x_hat_new)

C = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])
Cr = np.empty([2,4])
Cr = C
Cr[:,[1,2]] = Cr[:, [2,1]]
print("Cr: \n", Cr)
