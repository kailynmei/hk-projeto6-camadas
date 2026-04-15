
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# 1. Parâmetros gerais do áudio
fs = 44100  # frequência de amostragem
duracao = 3  # tempo de gravação

# Dicionário de acordes (igual ao emissor)
acordes = {
    "Dó maior": [523.25, 659.25, 783.99],
    "Ré menor": [587.33, 698.46, 880.00],
    "Mi menor": [659.25, 783.99, 987.77],
    "Fá maior": [698.46, 880.00, 1046.50],
    "Sol maior": [783.99, 987.77, 1174.66],
    "Lá menor": [880.00, 1046.50, 1318.51],
    "Si menor 5b": [493.88, 587.33, 698.46]
}

# 2. GRAVAÇÃO DO SINAL
print("Gravando áudio...")
audio = sd.rec(int(fs * duracao), samplerate=fs, channels=1)
sd.wait()
print("Gravação finalizada!")

# transforma em vetor
audio = audio.flatten()

# normaliza o sinal (melhora análise)
audio = audio / np.max(np.abs(audio))

# 3. GRÁFICO NO TEMPO
amostrasPlot = 1000

t = np.linspace(0, duracao, int(fs * duracao), endpoint=False)

plt.figure(figsize=(10, 4))
plt.plot(t[:amostrasPlot], audio[:amostrasPlot])
plt.title("Sinal recebido no tempo")
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

# 4. FFT (domínio da frequência)
N = len(audio)
fftVals = np.fft.fft(audio)
freqs = np.fft.fftfreq(N, d=1/fs)

metade = N // 2
freqsPos = freqs[:metade]
magnitudes = np.abs(fftVals[:metade]) / N

# 5. DETECÇÃO DE PICOS
# encontra picos relevantes no espectro
picos, _ = find_peaks(magnitudes, height=0.01)

freqs_detectadas = freqsPos[picos]

print("\nFrequências detectadas:")
print(freqs_detectadas[:10])

# 6. IDENTIFICAÇÃO DO ACORDE
def identifica_acorde(freqs_detectadas):
    for nome, freqs in acordes.items():
        match = 0

        for f in freqs:
            for fd in freqs_detectadas:
                if abs(f - fd) < 20:  # tolerância
                    match += 1

        if match >= 3:
            return nome

    return "Desconhecido"

acorde = identifica_acorde(freqs_detectadas)

print(f"\nAcorde identificado: {acorde}")

# 7. GRÁFICO DA FREQUÊNCIA
plt.figure(figsize=(10, 4))
plt.plot(freqsPos, magnitudes)
plt.plot(freqsPos[picos], magnitudes[picos], "x")  # marca os picos
plt.title("Espectro de frequência (recebido)")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 2000)
plt.grid()
plt.show()
