import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

# 1. Parâmetros gerais do áudio
fs = 44100 # Freq. de amostragem: escolhemos esse valor por ser um valor padrão
# de áudio digital e suficientemente alto p representar corretamente 
# as freqs. dos acordes da tabela

duracao = 3 # qnt tempo o acorde vai tocar (em segundos)

# Dicionário de acordes (enunciado)
acordes = {
    "1": ("Dó maior", [523.25, 659.25, 783.99]),
    "2": ("Ré menor", [587.33, 698.46, 880.00]),
    "3": ("Mi menor", [659.25, 783.99, 987.77]),
    "4": ("Fá maior", [698.46, 880.00, 1046.50]),
    "5": ("Sol maior", [783.99, 987.77, 1174.66]),
    "6": ("Lá menor", [880.00, 1046.50, 1318.51]),
    "7": ("Si menor 5b", [493.88, 587.33, 698.46])
}


# 2. Escolha do acorde pelo usuário
print("Escolha o acorde:")
for chave, (nome, freqs) in acordes.items():
    print(f'{chave} - {nome}: {freqs}')

opcao = input("Digite o número do acorde:")
if opcao not in acordes:
    print("Opção inválida")
    exit()

nomeAcorde, frequencias = acordes[opcao]
f1, f2, f3 = frequencias

# 3. Gerando o sinal no tempo
t = np.linspace(0, duracao, int(fs * duracao), endpoint = False) # vetor de tempo
# gerando as 3 senóides
s1 = np.sin(2 * np.pi * f1 * t)
s2 = np.sin(2 * np.pi * f2 * t)
s3 = np.sin(2 * np.pi * f3 * t)
tone = s1 + s2 + s3 # a variavel tone contém as senoides a serem executadas, que somadas 

# 4. Normalização e reprodução do áudio
# como a soma de 3 senóides pode ultrapassar o intervalo usual de a plitudoe,
# achamos melhor normalizar
tone = tone / np.max(np.abs(tone))
sd.play(tone, fs) # tocar o áudio
sd.wait()

# 5. Gráfico no domínio do tempo
amostrasPlot = 1000
plt.figure(figsize=(10, 4))
plt.plot(t[:amostrasPlot], tone[:amostrasPlot], color="pink", label="Sinal emitido")
plt.title(f"Sinal no tempo - {nomeAcorde}")
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

# 6. Gráfico no domínio da frequência
# o gráfico da FFT mostra quais freqs compõem o sinal e 
# com que amplitude elas aparecem
N = len(tone)
fftVals = np.fft.fft(tone)
freqs = np.fft.fftfreq(N, d=1/fs)
metade = N // 2
freqsPos = freqs[:metade]
magnitudes = np.abs(fftVals[:metade]) / N

plt.figure(figsize=(10, 4))
plt.plot(freqsPos, magnitudes, color="hotpink", label="FFT do sinal")
plt.title(f"Espectro de frequência - {nomeAcorde}")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 2000)
plt.legend()
plt.grid()

plt.show() # colocamos só 1 p mostrar os 2 juntos