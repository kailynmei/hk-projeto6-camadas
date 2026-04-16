import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

# 1. Parâmetros gerais do áudio
# Freq. de amostragem: define quantas amostras por segundo sao capturadas ou reproduzidas
# pelo teorema de Nyquist, fs deve ser pelo menos o dobro da maior freq q queremos representar
fs = 44100 # escolhemos esse valor por ser um valor padrão
# de áudio digital e suficientemente alto p representar corretamente 
# as freqs. dos acordes da tabela

duracao = 3 # qnt tempo o acorde vai tocar (em segundos)

# Dicionário de acordes (enunciado)
# 3 notas simultâneas somadas formam o sinal emitido
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
# O vetor de tempo t é construído com int(fs * duracao) pontos igualmente
# espaçados de 0 até 'duracao' segundos. 
# O espaçamento entre pontos consecutivos é 1/fs (período de amostragem Ts), 
# ou seja, estamos discretizando o tempo 
t = np.linspace(0, duracao, int(fs * duracao), endpoint = False) # vetor de tempo

# Cada nota do acorde é uma senoide pura: x(t) = sen(2π·f·t)
# A multiplicação 2π·f converte frequência (Hz) em frequência angular (rad/s),
# que é o ω da fórmula da Transformada de Fourier: X(ω) = ∫x(t)e^(-jωt)dt
s1 = np.sin(2 * np.pi * f1 * t)
s2 = np.sin(2 * np.pi * f2 * t)
s3 = np.sin(2 * np.pi * f3 * t)

# construímos o sinal a partir das freqs.
# x(t)
tone = s1 + s2 + s3 

# 4. Normalização e reprodução do áudio
# A soma de 3 senoides pode ter amplitude máxima de até 3 (quando as 3
# estão em fase - se as 3 estao no pico, soma = 3). 

# Placas de áudio esperam sinais no intervalo [-1, +1],
# então dividimos pelo valor absoluto máximo (normalização).
# A forma do sinal (e portanto seu conteúdo em frequência) não muda,
# apenas a escala da amplitude.
# a soma, qualquer que seja, fica sempre entre
tone = tone / np.max(np.abs(tone))
sd.play(tone, fs) # envia o sinal discreto p placa de áudio, q faz a conversao
# Digital → Analógico, reproduzimos as senoides como pressão sonora.
sd.wait() # bloq o programa até o áudio terminar

# 5. Gráfico no domínio do tempo
# plotamos apenas as primeiras 1000 amostras p visualizar a forma de onda com clareza
amostrasPlot = 1000
plt.figure(figsize=(10, 4))
plt.plot(t[:amostrasPlot], tone[:amostrasPlot], color="pink", label="Sinal emitido")
plt.title(f"Sinal no tempo - {nomeAcorde}")
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

# 6. Gráfico no domínio da frequência
# a transformada de Fourier recebe o sinal no dominio do tempo x(t) e 
# retorna X(ω): para cada frequência, um número complexo
# cuja magnitude indica "quanto" aquela frequência está presente no sinal.

# A fórmula contínua é:  X(ω) = ∫ x(t) · e^(-jωt) dt
# Na versão discreta (DFT), a integral vira somatório sobre as N amostras.

N = len(tone) # número total de amostras = fs * duracao = 132300

fftVals = np.fft.fft(tone) # np.fft.fft retorna N números complexos. 
# Cada posição k corresponde a uma frequência específica

freqs = np.fft.fftfreq(N, d=1/fs) # fftfreq gera o vetor de frequências correspondente a cada posição da FFT.
# d=1/fs define o espaçamento temporal entre amostras (período de amostragem Ts).
# A resolução em frequência é fs/N = 44100/132300 ≈ 0.33 Hz 

# Pegamos apenas a metade positiva do espectro (frequências 0 até fs/2).
# fs/2 é a frequência de Nyquist — limite máximo representável com essa fs.
metade = N // 2 

freqsPos = freqs[:metade]

# np.abs() calcula o módulo do número complexo X(ω) = √(real² + imag²).
# Isso nos dá a MAGNITUDE de cada frequência — o que plotamos no gráfico.
# Dividimos por N para normalizar a amplitude
magnitudes = np.abs(fftVals[:metade]) / N

plt.figure(figsize=(10, 4))
plt.plot(freqsPos, magnitudes, color="hotpink", label="FFT do sinal")
plt.title(f"Espectro de frequência - {nomeAcorde}")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 2000) # limitamos a 2000 Hz pq os acordes ficam todos abaixo disso
plt.legend()
plt.grid()

plt.show() # colocamos só 1 p mostrar os 2 juntos