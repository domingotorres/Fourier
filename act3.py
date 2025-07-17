import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Parámetros de la señal
fs = 1000  # Frecuencia de muestreo (Hz)
T = 1      # Duración de la señal (segundos)
t = np.linspace(0, T, int(fs * T), endpoint=False) # Vector de tiempo

# Señal compuesta: suma de dos senos con diferentes frecuencias
f1 = 50   # Frecuencia 1 (Hz)
f2 = 200  # Frecuencia 2 (Hz)
signal_pure = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)

# Ruido blanco gaussiano
noise = 0.5 * np.random.randn(len(t))

# Señal con ruido
signal_noisy = signal_pure + noise

# Graficar las señales antes de aplicar el filtro
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(t, signal_pure)
plt.title('Señal Pura (50 Hz y 200 Hz)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, noise)
plt.title('Ruido Blanco Gaussiano')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, signal_noisy)
plt.title('Señal con Ruido')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.grid(True)

plt.tight_layout()
plt.show()

#