import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from mpl_toolkits.mplot3d import Axes3D

# Adatok beolvasása és konvertálás
df = pd.read_csv('exp_2_imu.csv', names=['timestamp', 'AccelX', 'AccelY', 'AccelZ', 
                                          'GyroX', 'GyroY', 'GyroZ', 
                                          'MagX', 'MagY', 'MagZ'])

# Összes oszlop numerikussá alakítása
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna()

# Idő számítása
df['time'] = df['timestamp'] - df['timestamp'].iloc[0]
if df['time'].max() > 1000:
    df['time'] = df['time'] / 1000

dt = np.mean(np.diff(df['time']))

# Egyszerű integrálás szűrés nélkül
# Először integráljuk a gyorsulást hogy megkapjuk a sebességet
velocity_x = integrate.cumulative_trapezoid(df['AccelX'].values, dx=dt, initial=0)
velocity_y = integrate.cumulative_trapezoid(df['AccelY'].values, dx=dt, initial=0)
velocity_z = integrate.cumulative_trapezoid(df['AccelZ'].values, dx=dt, initial=0)

# A sebességből kivonjuk a lineáris trendet (drift csökkentés)
t = df['time'].values
velocity_x = velocity_x - np.polyval(np.polyfit(t, velocity_x, 1), t)
velocity_y = velocity_y - np.polyval(np.polyfit(t, velocity_y, 1), t)
velocity_z = velocity_z - np.polyval(np.polyfit(t, velocity_z, 1), t)

# Második integrálás a pozícióhoz
position_x = integrate.cumulative_trapezoid(velocity_x, dx=dt, initial=0)
position_y = integrate.cumulative_trapezoid(velocity_y, dx=dt, initial=0)
position_z = integrate.cumulative_trapezoid(velocity_z, dx=dt, initial=0)

# 3D ábra
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Pálya rajzolása színátmenettel
points = ax.scatter(position_x, position_y, position_z, 
                   c=range(len(position_x)), cmap='viridis', s=5)

# Start és végpont kiemelése
ax.scatter(position_x[0], position_y[0], position_z[0], 
          c='green', s=100, label='Start')
ax.scatter(position_x[-1], position_y[-1], position_z[-1], 
          c='red', s=100, label='End')

# Vonal a pálya mentén
ax.plot(position_x, position_y, position_z, 'b-', alpha=0.3, linewidth=1)

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('IMU Mozgáspálya')
ax.legend()

# Színskála hozzáadása
plt.colorbar(points, ax=ax, label='Minta sorszáma')

plt.show()

# XY sík nézet
plt.figure(figsize=(10, 8))
plt.plot(position_x, position_y, 'b-', linewidth=1)
plt.scatter(position_x[0], position_y[0], c='green', s=100, label='Start')
plt.scatter(position_x[-1], position_y[-1], c='red', s=100, label='End')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('XY sík - Felülnézet')
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()