import numpy as np
import matplotlib.pyplot as plt
# Parámetros comunes
fs = 1000  # Frecuencia de muestreo (Hz)
t = np.linspace(0, 1, fs, endpoint=False)  # Vector de tiempo de 0 a 1 segundo

# --- Pulso Rectangular ---
duration_rect = 0.2
start_time_rect = 0.3
rect_pulse = np.zeros_like(t)
rect_pulse[(t >= start_time_rect) & (t < start_time_rect + duration_rect)] = 1

plt.figure(figsize=(10, 6))  # Tamaño de la figura
plt.plot(t, rect_pulse, linewidth=2, color='blue')  # Línea más gruesa y color
plt.title('Pulso Rectangular en el Dominio del Tiempo', fontsize=16, fontweight='bold')
plt.xlabel('Tiempo (s)', fontsize=12)
plt.ylabel('Amplitud', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)  # Cuadrícula tenue
plt.xlim(0, 1)  # Limitar el eje x
plt.ylim(-0.1, 1.2) # Limitar el eje y
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()  # Ajustar espaciado para evitar etiquetas cortadas
plt.show()

# --- Función Escalón ---
step_time = 0.5
unit_step = np.zeros_like(t)
unit_step[(t >= step_time)] = 1

plt.figure(figsize=(10, 6))
plt.plot(t, unit_step, linewidth=2, color='green')
plt.title('Función Escalón Unitario en el Dominio del Tiempo', fontsize=16, fontweight='bold')
plt.xlabel('Tiempo (s)', fontsize=12)
plt.ylabel('Amplitud', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(0, 1)
plt.ylim(-0.1, 1.2)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()

# --- Función Senoidal ---
frequency = 5  # Hz
amplitude = 1
sin_wave = amplitude * np.sin(2 * np.pi * frequency * t)

plt.figure(figsize=(10, 6))
plt.plot(t, sin_wave, linewidth=2, color='red')
plt.title('Función Senoidal en el Dominio del Tiempo (f = {} Hz)'.format(frequency), fontsize=16, fontweight='bold')
plt.xlabel('Tiempo (s)', fontsize=12)
plt.ylabel('Amplitud', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(0, 1)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()