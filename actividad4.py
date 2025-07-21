import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# --- 1. Configuración Inicial y Definición de Parámetros ---
print("--- 1. Configuración de Parámetros ---")
fs = 100000  # Frecuencia de muestreo (Hz) - Debe ser al menos 2 * (fc + fm) para evitar aliasing
T = 1 / fs   # Periodo de muestreo
t = np.arange(0, 1, T) # Vector de tiempo de 1 segundo

# Parámetros de la señal del mensaje (información a transmitir)
Am = 1      # Amplitud de la señal del mensaje
fm = 10     # Frecuencia de la señal del mensaje (Hz) - Baja frecuencia
print(f"Señal de Mensaje: Amplitud={Am}, Frecuencia={fm} Hz")

# Parámetros de la señal portadora
Ac = 1      # Amplitud de la señal portadora (puede ser 1 si se ajusta mu)
fc = 1000   # Frecuencia de la señal portadora (Hz) - Alta frecuencia
print(f"Señal Portadora: Amplitud={Ac}, Frecuencia={fc} Hz")

# Índice de modulación (para AM de doble banda lateral con portadora)
# mu = (A_max - A_min) / (A_max + A_min)
# mu debe ser <= 1 para evitar la sobremodulación y distorsión en la envolvente.
mu = 0.8 # Índice de modulación (0 < mu <= 1)
print(f"Índice de Modulación (mu): {mu}\n")

# --- 2. Creación de Señales ---
print("--- 2. Creación de Señales ---")
# Señal del mensaje (m(t)) - una señal coseno como ejemplo de información
m_t = Am * np.cos(2 * np.pi * fm * t)
print("Señal del mensaje (m(t)) creada.")

# Señal portadora (c(t)) - la onda de alta frecuencia
c_t = Ac * np.cos(2 * np.pi * fc * t)
print("Señal portadora (c(t)) creada.\n")

# --- 3. Graficar la Señal de Entrada (Mensaje) antes de la Modulación ---
print("--- 3. Visualización de la Señal del Mensaje ---")
plt.figure(figsize=(12, 6))
plt.plot(t, m_t, label='Señal del Mensaje $m(t)$')
plt.title('Señal del Mensaje en el Dominio del Tiempo')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.grid(True)
plt.xlim(0, 0.2 / fm) # Mostrar unas pocas oscilaciones de la señal del mensaje
plt.legend()
plt.tight_layout()
plt.show()
print("Gráfica de la señal del mensaje en el tiempo generada.\n")

# --- 4. Implementación de la Modulación en Amplitud (AM) ---
print("--- 4. Implementación de Modulación AM ---")
# Fórmula para la señal modulada AM (Double Sideband - With Carrier, DSB-WC)
# s(t) = Ac * (1 + mu * m(t)/Am) * cos(2 * pi * fc * t)
# donde m(t)/Am normaliza el mensaje a un rango de -1 a 1.
s_t = Ac * (1 + mu * m_t / Am) * c_t
print("Señal modulada AM (s(t)) creada.\n")

# --- 5. Visualización de la Señal Modulada en el Dominio del Tiempo y la Frecuencia ---

# Dominio del Tiempo
print("--- 5.1. Visualización de Señal Modulada en el Tiempo ---")
plt.figure(figsize=(12, 7))
plt.plot(t, s_t, label='Señal Modulada AM $s(t)$')
plt.plot(t, Ac * (1 + mu * m_t / Am), 'r--', label='Envolvente Superior $A_c(1 + \mu m(t)/A_m)$')
plt.plot(t, -Ac * (1 + mu * m_t / Am), 'r--', label='Envolvente Inferior $-A_c(1 + \mu m(t)/A_m)$')
plt.title('Señal Modulada en Amplitud en el Dominio del Tiempo')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.grid(True)
plt.xlim(0, 0.05) # Mostrar unas pocas oscilaciones de la portadora para ver la envolvente
plt.legend()
plt.tight_layout()
plt.show()
print("Gráfica de la señal modulada en el tiempo con envolventes generada.\n")

# Dominio de la Frecuencia (Análisis de Fourier)
print("--- 5.2. Visualización de Señal Modulada en la Frecuencia (Espectro) ---")
N = len(s_t) # Número de puntos en la señal
yf = fft(s_t) # Realizar la Transformada Rápida de Fourier
xf = fftfreq(N, T)[:N//2] # Calcular las frecuencias correspondientes (solo el lado positivo)

plt.figure(figsize=(12, 6))
# Se multiplica por 2.0/N para normalizar la magnitud del espectro
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.title('Espectro de Magnitud de la Señal Modulada AM')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud Normalizada')
plt.grid(True)
plt.xlim(0, fc + fm + 200) # Limitar el eje X para ver la portadora y las bandas laterales
plt.axvline(fc, color='green', linestyle=':', label='Frecuencia Portadora ($f_c$)')
plt.axvline(fc + fm, color='purple', linestyle=':', label='Banda Lateral Superior ($f_c + f_m$)')
plt.axvline(fc - fm, color='orange', linestyle=':', label='Banda Lateral Inferior ($f_c - f_m$)')
plt.legend()
plt.tight_layout()
plt.show()
print("Gráfica del espectro de la señal modulada en frecuencia generada.\n")

# --- 6. Análisis de Impacto: Ruido, Atenuación y Distorsión ---

# --- 6.1. Introducir Ruido y Observar Cómo Afecta la Señal ---
print("--- 6.1. Análisis de Ruido ---")
# Generar ruido gaussiano aditivo blanco (AWGN)
# La potencia del ruido (varianza) determina la intensidad del ruido.
potencia_ruido = 0.5 # Ajustar este valor (ej: 0.1 para poco ruido, 1.0 para mucho)
# np.random.normal(media, desviación_estándar, forma)
# sqrt(potencia_ruido) es la desviación estándar
ruido = np.random.normal(0, np.sqrt(potencia_ruido), s_t.shape)
s_t_ruido = s_t + ruido
print(f"Ruido gaussiano con potencia {potencia_ruido} introducido.")

# Graficar la señal con ruido en el dominio del tiempo
plt.figure(figsize=(12, 6))
plt.plot(t, s_t_ruido, label='Señal Modulada AM con Ruido')
plt.title('Señal Modulada en Amplitud con Ruido en el Dominio del Tiempo')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.grid(True)
plt.xlim(0, 0.05)
plt.legend()
plt.tight_layout()
plt.show()
print("Gráfica de la señal AM con ruido en el tiempo generada.")

# Espectro de la señal con ruido
yf_ruido = fft(s_t_ruido)
plt.figure(figsize=(12, 6))
plt.plot(xf, 2.0/N * np.abs(yf_ruido[0:N//2]))
plt.title('Espectro de Magnitud de la Señal Modulada AM con Ruido')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud Normalizada')
plt.grid(True)
plt.xlim(0, fc + fm + 200) # Mantener el mismo rango de frecuencia
plt.axvline(fc, color='green', linestyle=':', label='Frecuencia Portadora ($f_c$)')
plt.axvline(fc + fm, color='purple', linestyle=':', label='Banda Lateral Superior ($f_c + f_m$)')
plt.axvline(fc - fm, color='orange', linestyle=':', label='Banda Lateral Inferior ($f_c - f_m$)')
plt.legend()
plt.tight_layout()
plt.show()
print("Gráfica del espectro de la señal AM con ruido en frecuencia generada.\n")
print("Observación del Ruido: El ruido en el dominio del tiempo se ve como variaciones aleatorias sobre la señal. En el dominio de la frecuencia, el ruido eleva el 'piso de ruido' en todo el espectro, dificultando la distinción de las componentes de la señal.\n")


# --- 6.2. Análisis de Atenuación ---
print("--- 6.2. Análisis de Atenuación ---")
# Simulación de atenuación: se reduce la amplitud de la señal.
factor_atenuacion = 0.3 # Un factor entre 0 y 1 (ej: 0.3 para una atenuación del 70%)
s_t_atenuada = s_t * factor_atenuacion
print(f"Señal atenuada por un factor de {factor_atenuacion} creada.")

# Graficar la señal atenuada en el dominio del tiempo
plt.figure(figsize=(12, 6))
plt.plot(t, s_t_atenuada, label='Señal Modulada AM Atenuada')
plt.title('Señal Modulada en Amplitud Atenuada en el Dominio del Tiempo')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.grid(True)
plt.xlim(0, 0.05)
plt.legend()
plt.tight_layout()
plt.show()
print("Gráfica de la señal AM atenuada en el tiempo generada.")

# Espectro de la señal atenuada
yf_atenuada = fft(s_t_atenuada)
plt.figure(figsize=(12, 6))
plt.plot(xf, 2.0/N * np.abs(yf_atenuada[0:N//2]))
plt.title('Espectro de Magnitud de la Señal Modulada AM Atenuada')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud Normalizada')
plt.grid(True)
plt.xlim(0, fc + fm + 200)
plt.axvline(fc, color='green', linestyle=':', label='Frecuencia Portadora ($f_c$)')
plt.axvline(fc + fm, color='purple', linestyle=':', label='Banda Lateral Superior ($f_c + f_m$)')
plt.axvline(fc - fm, color='orange', linestyle=':', label='Banda Lateral Inferior ($f_c - f_m$)')
plt.legend()
plt.tight_layout()
plt.show()
print("Gráfica del espectro de la señal AM atenuada en frecuencia generada.\n")
print("Observación de Atenuación: La atenuación reduce la amplitud total de la señal en el dominio del tiempo y, consecuentemente, la magnitud de todas las componentes de frecuencia en el espectro. Esto es como bajar el volumen de la señal.\n")

# --- 6.3. Análisis de Distorsión (Ejemplo Simple de Distorsión No Lineal) ---
# La distorsión es un tema amplio. Aquí simulamos una distorsión no lineal simple (ej. por un amplificador saturado)
print("--- 6.3. Análisis de Distorsión ---")
# Simulación de distorsión no lineal: agregamos un término cuadrático a la señal modulada
# Esto introduce armónicos y componentes de intermodulación
# Coeficiente de distorsión (ajustar para mayor o menor distorsión)
k_distorsion = 0.2
s_t_distorsion = s_t + k_distorsion * (s_t**2) # Ejemplo de distorsión cuadrática
print(f"Distorsión no lineal (cuadrática con k={k_distorsion}) introducida.")

# Graficar la señal distorsionada en el dominio del tiempo
plt.figure(figsize=(12, 6))
plt.plot(t, s_t_distorsion, label='Señal Modulada AM Distorsionada')
plt.title('Señal Modulada en Amplitud Distorsionada en el Dominio del Tiempo')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.grid(True)
plt.xlim(0, 0.05)
plt.legend()
plt.tight_layout()
plt.show()
print("Gráfica de la señal AM distorsionada en el tiempo generada.")

# Espectro de la señal distorsionada
yf_distorsion = fft(s_t_distorsion)
plt.figure(figsize=(12, 6))
plt.plot(xf, 2.0/N * np.abs(yf_distorsion[0:N//2]))
plt.title('Espectro de Magnitud de la Señal Modulada AM Distorsionada')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud Normalizada')
plt.grid(True)
# Ampliar el rango X para ver posibles nuevos armónicos de la distorsión
plt.xlim(0, 3 * fc + 3 * fm)
plt.axvline(fc, color='green', linestyle=':', label='Frecuencia Portadora ($f_c$)')
plt.axvline(fc + fm, color='purple', linestyle=':', label='Banda Lateral Superior ($f_c + f_m$)')
plt.axvline(fc - fm, color='orange', linestyle=':', label='Banda Lateral Inferior ($f_c - f_m$)')
plt.legend()
plt.tight_layout()
plt.show()
print("Gráfica del espectro de la señal AM distorsionada en frecuencia generada.\n")
print("Observación de Distorsión: La distorsión altera la forma de onda de la señal en el dominio del tiempo, a menudo introduciendo picos o achatamientos. En el dominio de la frecuencia, la distorsión no lineal genera nuevas componentes de frecuencia (armónicos de la portadora y el mensaje, y componentes de intermodulación) que no estaban presentes en la señal original, lo que puede causar interferencia y degradación de la calidad de la señal.\n")

print("\n--- Fin del Análisis ---")