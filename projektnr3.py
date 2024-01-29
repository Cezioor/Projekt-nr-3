import numpy as np
import matplotlib.pyplot as plt

def Euler():
    h = 0.001
    t = np.arange(0, 30, h)

    N1 = np.zeros(len(t))
    N2 = np.zeros(len(t))

    N1[0] = n1_init
    N2[0] = n2_init

    for i in range(1, len(t)):
        N1[i] = N1[i - 1] + h * ((eps1 - gam1 * (h1 * N1[i - 1] + h2 * N2[i - 1])) * N1[i - 1])
        N2[i] = N2[i - 1] + h * ((eps2 - gam2 * (h1 * N1[i - 1] + h2 * N2[i - 1])) * N2[i - 1])
            
    return [N1, N2]


def main():
    # zadanie 1 ====
    K = 100000  
    r = .4   

    h = 0.01
    t = np.arange(75, 130, h) 

    Gx = np.zeros(len(t))
    Vx = np.zeros(len(t))

    Gx[0] = 10
    Vx[0] = 10

    for i in range(1, t.shape[0]):
        Gx[i] = Gx[i - 1] + h * r * Gx[i - 1] * np.log(K/Gx[i - 1])
        Vx[i] = Vx[i - 1] + h * r * Vx[i - 1] * (1 - Vx[i - 1]/K) 


    plt.plot(t, Gx, label='GOMPERTZ')
    plt.plot(t, Vx, label='VERHULST')
    plt.ylabel('ILOSC KIBICOW')
    plt.xlabel('CZAS')
    plt.title('POROWNANIE  GOMPERTZ I VERHULST')
    plt.legend()
    plt.show()  
    # zadanie 1 ====

    # zadanie 2 ====
    # a) ===========
    h = 0.001
    t = np.arange(0, 30, h)

    global eps1, eps2, gam1, gam2, h1, h2, n1_init, n2_init
    eps1, eps2, gam1, gam2, h1, h2, n1_init, n2_init = 1.25, .5, .5, .2, .1, .2, 3, 4

    an1, an2 = Euler()

    plt.plot(t, an1, label='N1 a')
    plt.plot(t, an2, label='N2 a')
    plt.ylabel('Populacja')
    plt.xlabel('czas')
    plt.title('Porownanie dwoch roznych wartosci')
    plt.legend()
    plt.show()
    # a) ===========

    # b) ===========
    eps1, eps2, gam1, gam2, h1, h2, n1_init, n2_init = 5, 5, 4, 8, 1, 4, 3, 4

    bn1, bn2 = Euler()

    plt.plot(t, bn1, label='N1 a')
    plt.plot(t, bn2, label='N2 a')
    plt.ylabel('Populacja')
    plt.xlabel('czas')
    plt.title('Porownanie dwoch roznych wartosci')
    plt.legend()
    plt.show()
    # b) ===========

    # c,d,e) =======
    h = 0.001
    t = np.arange(0, 20, h)

    eps1, eps2, gam1, gam2, h1, h2 = 0.8, 0.4, 1, 0.5, 0.3, 0.4
    n1_init, n2_init = 4, 8
    cn1, cn2 = Euler()

    n1_init, n2_init = 8, 8
    dn1, dn2 = Euler()

    n1_init, n2_init = 12, 8
    en1, en2 = Euler()

    plt.plot(cn1, cn2)
    plt.plot(dn1, dn2)
    plt.plot(en1, en2)
    plt.show()

    # potret fazowy
    x = np.linspace(0, 12, 12)
    y = np.linspace(0, 12, 12)

    X, Y = np.meshgrid(x, y)
    dX = np.zeros(X.shape)
    dY = np.zeros(Y.shape)

    for i in range(X.shape[0]): 
        for j in range(Y.shape[0]):
            dX[i, j] = (eps1 - gam1 * (h1 * X[i, j] + h2 * Y[i, j])) * X[i, j]
            dY[i, j] = (eps2 - gam2 * (h1 * X[i, j] + h2 * Y[i, j])) * Y[i, j]

    plt.quiver(X, Y, dX, dY)
    plt.plot(cn1, cn2, label='N1 = 4, N2 = 8')
    plt.plot(dn1, dn2, label='N1 = 8, N2 = 8')
    plt.plot(en1, en2, label='N1 = 12, N2 = 8')
    plt.legend()
    plt.xlabel('Populacja N1')
    plt.ylabel('Populacja N2')
    plt.title('Graf z trzema roznymi krzywymi')
    plt.show()

main()
