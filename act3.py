import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# --- 1. Definición de la señal de entrada ---
# Parámetros de muestreo y duración
fs = 1000  # Frecuencia de muestreo (Hz)
T = 1      # Duración de la señal (segundos)
t = np.linspace(0, T, int(fs * T), endpoint=False) # Vector de tiempo

# Crear una señal compuesta: suma de dos senos con diferentes frecuencias
f1_pure = 50   # Frecuencia deseada (Hz)
f2_pure = 200  # Frecuencia no deseada (Hz)
signal_pure = np.sin(2 * np.pi * f1_pure * t) + 0.5 * np.sin(2 * np.pi * f2_pure * t)

# Añadir ruido blanco gaussiano para simular una señal real
noise_amplitude = 0.5
noise = noise_amplitude * np.random.randn(len(t))
signal_noisy = signal_pure + noise

# --- 2. Graficar las señales antes de aplicar el filtro ---
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, signal_pure)
plt.title(f'Señal Pura ({f1_pure} Hz y {f2_pure} Hz)')
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
plt.title('Señal Compuesta con Ruido')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.grid(True)

plt.tight_layout()
plt.show()

# --- Parámetros comunes para el diseño de filtros ---
nyquist = 0.5 * fs # Frecuencia de Nyquist

# --- 3. y 4. Implementación de Filtros IIR (Butterworth) ---

## Filtro Pasa Bajos (Low-Pass Filter)
cutoff_lp = 70 # Frecuencia de corte para el pasa bajos (Hz)
order_lp = 5   # Orden del filtro

# Diseño del filtro Butterworth IIR
b_lp, a_lp = signal.butter(order_lp, cutoff_lp / nyquist, btype='low', analog=False)

# Respuesta en frecuencia del filtro
w_lp, h_lp = signal.freqz(b_lp, a_lp, worN=8000)
freq_lp = w_lp * fs / (2 * np.pi)

# Aplicar el filtro a la señal con ruido
signal_filtered_lp = signal.lfilter(b_lp, a_lp, signal_noisy)

# Graficar resultados del filtro pasa bajos
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, signal_noisy)
plt.title('Señal Original con Ruido (Pasa Bajos)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(freq_lp, 20 * np.log10(abs(h_lp)))
plt.title(f'Respuesta en Frecuencia del Filtro Pasa Bajos Butterworth (Orden {order_lp}, Fc={cutoff_lp} Hz)')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Ganancia (dB)')
plt.axvline(cutoff_lp, color='red', linestyle='--', label='Frecuencia de Corte')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t, signal_filtered_lp)
plt.title('Señal Filtrada (Pasa Bajos)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.grid(True)

plt.tight_layout()
plt.show()

## Filtro Pasa Altos (High-Pass Filter)
cutoff_hp = 150 # Frecuencia de corte para el pasa altos (Hz)
order_hp = 5    # Orden del filtro

# Diseño del filtro Butterworth IIR
b_hp, a_hp = signal.butter(order_hp, cutoff_hp / nyquist, btype='high', analog=False)

# Respuesta en frecuencia del filtro
w_hp, h_hp = signal.freqz(b_hp, a_hp, worN=8000)
freq_hp = w_hp * fs / (2 * np.pi)

# Aplicar el filtro a la señal con ruido
signal_filtered_hp = signal.lfilter(b_hp, a_hp, signal_noisy)

# Graficar resultados del filtro pasa altos
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, signal_noisy)
plt.title('Señal Original con Ruido (Pasa Altos)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(freq_hp, 20 * np.log10(abs(h_hp)))
plt.title(f'Respuesta en Frecuencia del Filtro Pasa Altos Butterworth (Orden {order_hp}, Fc={cutoff_hp} Hz)')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Ganancia (dB)')
plt.axvline(cutoff_hp, color='red', linestyle='--', label='Frecuencia de Corte')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t, signal_filtered_hp)
plt.title('Señal Filtrada (Pasa Altos)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.grid(True)

plt.tight_layout()
plt.show()

## Filtro Pasa Bandas (Band-Pass Filter)
lowcut_bp = 40  # Frecuencia de corte inferior (Hz)
highcut_bp = 60 # Frecuencia de corte superior (Hz)
order_bp = 5    # Orden del filtro

# Diseño del filtro Butterworth IIR
b_bp, a_bp = signal.butter(order_bp, [lowcut_bp / nyquist, highcut_bp / nyquist], btype='band', analog=False)

# Respuesta en frecuencia del filtro
w_bp, h_bp = signal.freqz(b_bp, a_bp, worN=8000)
freq_bp = w_bp * fs / (2 * np.pi)

# Aplicar el filtro a la señal con ruido
signal_filtered_bp = signal.lfilter(b_bp, a_bp, signal_noisy)

# Graficar resultados del filtro pasa bandas
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, signal_noisy)
plt.title('Señal Original con Ruido (Pasa Bandas)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(freq_bp, 20 * np.log10(abs(h_bp)))
plt.title(f'Respuesta en Frecuencia del Filtro Pasa Bandas Butterworth (Orden {order_bp}, {lowcut_bp}-{highcut_bp} Hz)')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Ganancia (dB)')
plt.axvline(lowcut_bp, color='red', linestyle='--', label='Fc Inferior')
plt.axvline(highcut_bp, color='red', linestyle='--', label='Fc Superior')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t, signal_filtered_bp)
plt.title('Señal Filtrada (Pasa Bandas)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.grid(True)

plt.tight_layout()
plt.show()

# --- 3. y 4. Implementación de Filtro FIR (con ventana) ---

## Filtro Pasa Bajos FIR (ejemplo para comparación)
cutoff_fir = 70 # Frecuencia de corte para el pasa bajos (Hz)
numtaps_fir = 61 # Número de taps (orden del filtro + 1). Preferiblemente impar para simetría.

# Diseño del filtro FIR con ventana Hamming
b_fir = signal.firwin(numtaps_fir, cutoff_fir / nyquist, window='hamming', pass_zero='lowpass')

# Respuesta en frecuencia del filtro
w_fir, h_fir = signal.freqz(b_fir, worN=8000)
freq_fir = w_fir * fs / (2 * np.pi)

# Aplicar el filtro a la señal con ruido
signal_filtered_fir = signal.lfilter(b_fir, 1.0, signal_noisy) # Para FIR, el coeficiente 'a' es 1

# Graficar resultados del filtro pasa bajos FIR
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, signal_noisy)
plt.title('Señal Original con Ruido (Pasa Bajos FIR)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(freq_fir, 20 * np.log10(abs(h_fir)))
plt.title(f'Respuesta en Frecuencia del Filtro Pasa Bajos FIR (Ventana Hamming, Taps={numtaps_fir}, Fc={cutoff_fir} Hz)')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Ganancia (dB)')
plt.axvline(cutoff_fir, color='red', linestyle='--', label='Frecuencia de Corte')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t, signal_filtered_fir)
plt.title('Señal Filtrada (Pasa Bajos FIR)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.grid(True)

plt.tight_layout()
plt.show()

# --- 6. Comparación de las señales antes y después del filtrado (Dominio de la Frecuencia) ---

# Calcular la FFT de la señal original con ruido
N = len(signal_noisy)
yf_noisy = np.fft.fft(signal_noisy)
xf = np.fft.fftfreq(N, 1 / fs)

# Calcular la FFT de la señal filtrada (Pasa Bajos IIR)
yf_filtered_lp = np.fft.fft(signal_filtered_lp)

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(xf[:N//2], 2/N * np.abs(yf_noisy[0:N//2]))
plt.title('Espectro de Frecuencia de la Señal con Ruido')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud')
plt.xlim(0, fs / 2)
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(xf[:N//2], 2/N * np.abs(yf_filtered_lp[0:N//2]))
plt.title(f'Espectro de Frecuencia de la Señal Filtrada (Pasa Bajos {cutoff_lp} Hz)')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud')
plt.xlim(0, fs / 2)
plt.grid(True)

plt.tight_layout()
plt.show()