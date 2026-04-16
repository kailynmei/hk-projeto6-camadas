import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# 1. Parâmetros gerais
fs = 44100
duracao = 3

# Dicionário de acordes (mesmo do emissor)
acordes = {
    "Dó maior":    [523.25, 659.25, 783.99],
    "Ré menor":    [587.33, 698.46, 880.00],
    "Mi menor":    [659.25, 783.99, 987.77],
    "Fá maior":    [698.46, 880.00, 1046.50],
    "Sol maior":   [783.99, 987.77, 1174.66],
    "Lá menor":    [880.00, 1046.50, 1318.51],
    "Si menor 5b": [493.88, 587.33, 698.46]
}

# 2. Gravação do áudio
print(f"Gravando por {duracao} segundos... 🎙️")
gravacao = sd.rec(int(fs * duracao), samplerate=fs, channels=1, dtype='float64')
sd.wait()
print("Gravação concluída!")

# 3. Tratar o sinal (gravação retorna lista de listas → achatar pra 1 lista)
sinal = gravacao.flatten()

# Ajuste de volume — descomente se o sinal estiver fraco
# sinal = sinal * 5.0

# 4. FFT do sinal gravado
N = len(sinal)
fftVals = np.fft.fft(sinal)
freqs = np.fft.fftfreq(N, d=1/fs)
metade = N // 2
freqsPos = freqs[:metade]
magnitudes = np.abs(fftVals[:metade]) / N

# 5. Identificar picos (mínimo 5, ajuste 'height' conforme o ambiente)
picos_idx, _ = find_peaks(magnitudes, height=0.001, distance=500)
freqs_pico = freqsPos[picos_idx]
mags_pico = magnitudes[picos_idx]

# Pegar os 5 maiores picos
top_n = 5
if len(picos_idx) >= top_n:
    maiores_idx = np.argsort(mags_pico)[-top_n:]
    freqs_detectadas = sorted(freqs_pico[maiores_idx])
else:
    freqs_detectadas = sorted(freqs_pico)

print(f"\nFrequências detectadas: {[round(f, 2) for f in freqs_detectadas]} Hz")

# 6. Identificar o acorde
tolerancia = 15  # margem de erro em Hz

def bate_frequencia(f_detectada, f_referencia):
    return abs(f_detectada - f_referencia) <= tolerancia

def checa_acorde(freqs_det, freqs_ref):
    # retorna True se as 3 frequências do acorde foram encontradas
    acertos = 0
    for f_ref in freqs_ref:
        for f_det in freqs_det:
            if bate_frequencia(f_det, f_ref):
                acertos += 1
                break
    return acertos >= 3

acorde_encontrado = None
for nome, freqs_ref in acordes.items():
    if checa_acorde(freqs_detectadas, freqs_ref):
        acorde_encontrado = nome
        break

if acorde_encontrado:
    print(f"\n✅ Acorde identificado: {acorde_encontrado}")
else:
    print("\n❌ Acorde não identificado. Tente ajustar 'height' no find_peaks ou o volume.")

# 7. Gráfico no domínio do tempo
amostrasPlot = 1000
t = np.linspace(0, duracao, N, endpoint=False)

plt.figure(figsize=(10, 4))
plt.plot(t[:amostrasPlot], sinal[:amostrasPlot], color="pink", label="Sinal recebido")
plt.title(f"Sinal recebido no tempo - {acorde_encontrado or 'não identificado'}")
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

# 8. Gráfico da FFT com picos marcados
plt.figure(figsize=(10, 4))
plt.plot(freqsPos, magnitudes, color="hotpink", label="FFT do sinal recebido")
plt.plot(freqs_pico, mags_pico, "x", color="deeppink", markersize=8, label="Picos detectados")
plt.title(f"Espectro de frequência recebido - {acorde_encontrado or 'não identificado'}")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 2000)
plt.legend()
plt.grid()

plt.show()  # mesmo esquema do emissor: 1 show pra mostrar os 2 juntos