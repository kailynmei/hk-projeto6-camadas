import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# PARÂMETROS GERAIS 
# Devem ser iguais aos do emissor para que a FFT opere na mesma escala de frequência
fs = 44100
duracao = 4

# Mesmo dicionário do emissor — usado para comparar as frequências detectadas
acordes = {
    "Dó maior":    [523.25, 659.25, 783.99],
    "Ré menor":    [587.33, 698.46, 880.00],
    "Mi menor":    [659.25, 783.99, 987.77],
    "Fá maior":    [698.46, 880.00, 1046.50],
    "Sol maior":   [783.99, 987.77, 1174.66],
    "Lá menor":    [880.00, 1046.50, 1318.51],
    "Si menor 5b": [493.88, 587.33, 698.46]
}

# GRAVAÇÃO 
# sd.rec captura fs*duracao amostras pelo microfone
# channels=1 = mono (um canal), dtype='float64' = precisão de ponto flutuante
print(f"Gravando por {duracao} segundos...")
gravacao = sd.rec(int(fs * duracao), samplerate=fs, channels=1, dtype='float64')
sd.wait()  # aguarda a gravação terminar
print("Gravação concluída!")

# TRATAMENTO DO SINAL
# sd.rec retorna array 2D (lista de listas) mesmo com 1 canal
# flatten() converte para array 1D, necessário para a FFT
sinal = gravacao.flatten()

# O microfone capta o som atenuado pelo ar — amplitude muito menor que o emissor
# Multiplicar amplifica o sinal para que os picos fiquem visíveis na FFT
# Esse valor pode precisar de ajuste dependendo da distância e volume
sinal = sinal * 50

# FFT DO SINAL GRAVADO 
N = len(sinal)
fftVals = np.fft.fft(sinal)
freqs = np.fft.fftfreq(N, d=1/fs)
metade = N // 2
freqsPos = freqs[:metade]
magnitudes = np.abs(fftVals[:metade]) / N

# FUNÇÃO DE PONTUAÇÃO 
# Margem de tolerância: a FFT tem resolução finita (fs/N Hz por bin)
# Com fs=44100 e N=194040, resolução ≈ 0.23 Hz — mas ruído desloca os picos levemente
# 10 Hz de tolerância é suficiente para absorver esse deslocamento sem confundir acordes
tolerancia = 10

def pontua_acorde(freqs_det, freqs_ref):
    """Conta quantas frequências de referência foram detectadas (máx. 3)"""
    acertos = 0
    for f_ref in freqs_ref:
        for f_det in freqs_det:
            if abs(f_det - f_ref) <= tolerancia:
                acertos += 1
                break  # encontrou esse f_ref, passa pro próximo
    return acertos

# DETECÇÃO DE PICOS 
# Filtramos abaixo de 450 Hz: todos os acordes têm frequências acima de 493 Hz
# Ruído de baixa frequência (ambiente, ventiladores, voz) ficaria abaixo disso
# Filtramos acima de 2000 Hz: nenhum acorde chega perto disso
mascara = (freqsPos > 450) & (freqsPos < 2000)
freqsPos_filtradas = freqsPos[mascara]
magnitudes_filtradas = magnitudes[mascara]

# find_peaks encontra máximos locais no espectro
# height: magnitude mínima para considerar um pico (filtra ruído de fundo)
# distance: mínimo de bins entre dois picos (evita detectar o mesmo pico duas vezes)
picos_idx, _ = find_peaks(magnitudes_filtradas, height=0.000001, distance=100)
freqs_pico = freqsPos_filtradas[picos_idx]
mags_pico = magnitudes_filtradas[picos_idx]

# Pega os 5 picos de maior magnitude
# As 3 frequências reais do acorde sempre têm energia muito maior que o ruído
# top_n=5 garante que capturamos as 3 certas sem incluir ruído demais
top_n = 5
if len(picos_idx) >= top_n:
    maiores_idx = np.argsort(mags_pico)[-top_n:]  # índices dos top_n maiores
    freqs_detectadas = sorted(freqs_pico[maiores_idx])
else:
    freqs_detectadas = sorted(freqs_pico)

print(f"Frequências detectadas: {[round(f, 2) for f in freqs_detectadas]} Hz")

# DEBUG: pontuação de cada acorde (útil para ajustes) 
print("\n--- Pontuações ---")
for nome, freqs_ref in acordes.items():
    p = pontua_acorde(freqs_detectadas, freqs_ref)
    print(f"{nome}: {p}/3")
print("------------------\n")

# IDENTIFICAÇÃO DO ACORDE 
# Percorre todos os acordes e escolhe o que tiver maior pontuação
# Isso evita empates onde o primeiro da lista sempre venceria
melhor_acorde = None
melhor_pontuacao = 0

for nome, freqs_ref in acordes.items():
    pontuacao = pontua_acorde(freqs_detectadas, freqs_ref)
    if pontuacao > melhor_pontuacao:
        melhor_pontuacao = pontuacao
        melhor_acorde = nome

# Só aceita como identificado se pelo menos 3 frequências bateram
acorde_encontrado = melhor_acorde if melhor_pontuacao >= 3 else None

if acorde_encontrado:
    print(f"Acorde identificado: {acorde_encontrado} ({melhor_pontuacao}/3)")
else:
    print("Acorde não identificado. Ajuste 'height' no find_peaks ou o volume.")

# GRÁFICO NO DOMÍNIO DO TEMPO 
# 200 amostras = ~0.005s, suficiente para ver os ciclos das senoides
amostrasPlot = 200
t = np.linspace(0, duracao, N, endpoint=False)

plt.figure(figsize=(10, 4))
plt.plot(t[:amostrasPlot], sinal[:amostrasPlot], color="pink", label="Sinal recebido")
plt.title(f"Sinal recebido no tempo - {acorde_encontrado or 'não identificado'}")
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

# GRÁFICO DA FFT COM PICOS COLORIDOS
# Separa os picos detectados em dois grupos:
# - picos do acorde: frequências que batem com o acorde identificado (X vermelho)
# - picos de ruído: demais frequências detectadas (X deeppink)
freqs_acorde_ref = list(acordes[acorde_encontrado]) if acorde_encontrado else []
picos_acorde_f, picos_acorde_m = [], []
picos_ruido_f, picos_ruido_m = [], []

for f, m in zip(freqs_pico, mags_pico):
    if any(abs(f - fref) <= tolerancia for fref in freqs_acorde_ref):
        picos_acorde_f.append(f)
        picos_acorde_m.append(m)
    else:
        picos_ruido_f.append(f)
        picos_ruido_m.append(m)

plt.figure(figsize=(10, 4))
plt.plot(freqsPos, magnitudes, color="hotpink", label="FFT do sinal recebido")
plt.plot(picos_ruido_f,  picos_ruido_m,  "x", color="deeppink", markersize=7,  label="Picos (ruído)")
plt.plot(picos_acorde_f, picos_acorde_m, "x", color="red",      markersize=12,
         markeredgewidth=2.5, label="Picos do acorde")
plt.title(f"Espectro de frequência recebido - {acorde_encontrado or 'não identificado'}")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 2000)
# ylim ignora o pico de ruído de baixa frequência para os picos do acorde ficarem visíveis
plt.ylim(0, magnitudes[freqsPos > 450].max() * 1.3)
plt.legend()
plt.grid()

plt.show()  # exibe os dois gráficos juntos