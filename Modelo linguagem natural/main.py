from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk import tokenize
import pandas as pd
import nltk
import seaborn as sns

import utils as u

df = u.df
bag_of_words = u.bag_of_words
regrassao_logistica = u.regrassao_logistica
vetorizar = u.vetorizar
matriz_esparsa_avaliacoes = u.matriz_esparsa_avaliacoes

todas_palavras = [texto for texto in df.avaliacao]
todas_palavras = ' '.join([texto for texto in df.avaliacao])

# Primeira amostra
# nuvem_palavras = WordCloud().generate(todas_palavras)
# plt.figure()
# plt.imshow(nuvem_palavras)
# plt.show()

# Definindo o tamanho da imagem
# nuvem_palavras = WordCloud(width=800, height=500, max_font_size=110).generate(todas_palavras)
# plt.figure(figsize=(10, 7))
# plt.imshow(nuvem_palavras, interpolation='bilinear')
# plt.axis('off')
# plt.show()

# Removendo palavras repetidas
# nuvem_palavras = WordCloud(width=800, height=500, max_font_size=110, collocations=False).generate(todas_palavras)
# plt.figure(figsize=(10, 7))
# plt.imshow(nuvem_palavras, interpolation='bilinear')
# plt.axis('off')
# plt.show()

# Aplicando função
# u.nuvem_palavras(df, 'avaliacao', 'negativo')
# u.nuvem_palavras(df, 'avaliacao', 'positivo')

# nltk.download('all')

# frases = ['um produto bom', 'um produto ruim']
# frequencia = nltk.FreqDist(frases)
# print(frequencia.most_common())

# Tokenização
# frase = 'O produto é excelente e a entrega foi muito rápida!'
token_espaco = tokenize.WhitespaceTokenizer()
# token_frase = token_espaco.tokenize(frase)
# print(token_frase)

token_frase = token_espaco.tokenize(todas_palavras)
frequencia = nltk.FreqDist(token_frase)
df_frequencia = pd.DataFrame({
    'Palavra': list(frequencia.keys()),
    'Frequência': list(frequencia.values())
})

# print(df_frequencia.head())

exemplo = df_frequencia.nlargest(columns='Frequência', n=20)

# print(exemplo)

# plt.figure(figsize=(20, 6))
# ax = sns.barplot(data=exemplo, x='Palavra', y='Frequência', color='gray')
# ax.set(ylabel='Contagem')
# plt.show()

# Removendo palavras irrelevantes
palavras_irrelevantes = nltk.corpus.stopwords.words('portuguese')

frase_processada = []
for opiniao in df.avaliacao:
    palavras_textos = token_espaco.tokenize(opiniao)
    nova_frase = [palavra for palavra in palavras_textos if palavra not in palavras_irrelevantes]
    frase_processada.append(' '.join(nova_frase))
    
df['tratamento_1'] = frase_processada

# print(df.head())

# u.classificar_texto(df, 'tratamento_1', 'sentimento') # Resultado de 81,09%

u.grafico_frequencia(df, 'tratamento_1', 20)