import matplotlib.pyplot as plt
import numpy as np
import time

# Pozíció / orientáció adatok tárolása
positions = []
orientations = []

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(-5, 5)
ax.set_xlabel('X (jobbra/balra)')
ax.set_ylabel('Y (fel/le)')
ax.set_zlabel('Z (előre/hátra)')
ax.set_title('3D Pozíció és Orientáció')

# Alapeset: 100 pontból egy mintapálya, változó kvaternióval
for i in range(100):
    # Folyamatos, példában növekvő pozíció
    positions.append([np.cos(i/25)*3, np.sin(i/25)*3, i*0.1-5])
    # Orientáció
    angle = i * np.pi/50
    w = np.cos(angle/2)
    x = np.sin(angle/2)
    y = np.sin(angle/2)*0.6
    z = np.sin(angle/2)*0.2
    orientations.append([w, x, y, z])

    # Rajz újrarenderelése
    ax.cla()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_xlabel('X (jobbra/balra)')
    ax.set_ylabel('Y (fel/le)')
    ax.set_zlabel('Z (előre/hátra)')
    ax.set_title('3D Pozíció és Orientáció')

    pos = np.array(positions)
    ax.plot(pos[:,0], pos[:,1], pos[:,2], 'b-')

    # Legutóbbi orientáció kirajzolása (kvaternióból mátrix)
    last_ori = orientations[-1]
    w, x, y, z = last_ori
    R = np.array([
        [1-2*(y**2+z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x**2 + y**2)]
    ])

    origin = pos[-1]
    scale = 0.5
    # Jobbra (X, piros), Fel (Y, zöld), Előre (Z, kék)
    ax.quiver(*origin, *(R[:,0]*scale), color='r')
    ax.quiver(*origin, *(R[:,1]*scale), color='g')
    ax.quiver(*origin, *(R[:,2]*scale), color='b')

    plt.pause(0.03)  # élő animáció

plt.show()
