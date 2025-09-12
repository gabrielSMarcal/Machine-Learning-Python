from wordcloud import WordCloud
import matplotlib.pyplot as plt

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
nuvem_palavras = WordCloud(width=800, height=500, max_font_size=110, collocations=False).generate(todas_palavras)
plt.figure(figsize=(10, 7))
plt.imshow(nuvem_palavras, interpolation='bilinear')
plt.axis('off')
plt.show()