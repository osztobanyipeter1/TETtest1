import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Adatok beolvasása
df = pd.read_csv('exp_2_imu.csv', names=['timestamp', 'AccelX', 'AccelY', 'AccelZ', 
                                          'GyroX', 'GyroY', 'GyroZ', 
                                          'MagX', 'MagY', 'MagZ'])

# Timestamp átalakítása numerikus értékké
# Feltételezzük, hogy ez lehet másodperc vagy milliszekundum
df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')

# Ellenőrizzük, hogy van-e NaN érték
if df['timestamp'].isna().any():
    print("Figyelem: Néhány timestamp érték nem konvertálható számmá!")
    # Eldobjuk a NaN sorokat
    df = df.dropna(subset=['timestamp'])

# Idő normalizálása (az első értékhez képest)
df['time'] = df['timestamp'] - df['timestamp'].iloc[0]

# Ha nagyon nagy számok (pl. milliszekundum), átváltjuk másodpercre
if df['time'].max() > 1000:
    df['time'] = df['time'] / 1000  # ms -> s
    print("Idő átváltva másodpercre (feltételezve, hogy ms-ben volt)")

# Ábra létrehozása
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Gyorsulásmérő
axes[0].plot(df['time'], df['AccelX'], label='Accel X', color='red', linewidth=1)
axes[0].plot(df['time'], df['AccelY'], label='Accel Y', color='green', linewidth=1)
axes[0].plot(df['time'], df['AccelZ'], label='Accel Z', color='blue', linewidth=1)
axes[0].set_ylabel('Gyorsulás (m/s²)')
axes[0].set_title('Gyorsulásmérő adatok')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Giroszkóp
axes[1].plot(df['time'], df['GyroX'], label='Gyro X', color='red', linewidth=1)
axes[1].plot(df['time'], df['GyroY'], label='Gyro Y', color='green', linewidth=1)
axes[1].plot(df['time'], df['GyroZ'], label='Gyro Z', color='blue', linewidth=1)
axes[1].set_ylabel('Szögsebesség (rad/s)')
axes[1].set_title('Giroszkóp adatok')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Magnetometer
axes[2].plot(df['time'], df['MagX'], label='Mag X', color='red', linewidth=1)
axes[2].plot(df['time'], df['MagY'], label='Mag Y', color='green', linewidth=1)
axes[2].plot(df['time'], df['MagZ'], label='Mag Z', color='blue', linewidth=1)
axes[2].set_xlabel('Idő (s)')
axes[2].set_ylabel('Mágneses tér (µT)')
axes[2].set_title('Magnetometer adatok')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Adatok statisztikáinak kiírása
print("\nAdatok statisztikái:")
print(f"Összes minta: {len(df)}")
print(f"Időtartam: {df['time'].max():.2f} másodperc")
print(f"Mintavételi frekvencia: {len(df)/df['time'].max():.1f} Hz")