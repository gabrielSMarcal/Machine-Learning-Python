import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
import unidecode
from nltk import tokenize, ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud

from utils import df, regrassao_logistica, SEED, classificar_texto, grafico_frequencia

todas_palavras = [texto for texto in df.avaliacao]
todas_palavras = ' '.join([texto for texto in df.avaliacao])

'''PRIMEIRA AMOSTRA'''
# nuvem_palavras = WordCloud().generate(todas_palavras)
# plt.figure()
# plt.imshow(nuvem_palavras)
# plt.show()

'''DEFININDO O TAMANHO DA IMAGEM'''
# nuvem_palavras = WordCloud(width=800, height=500, max_font_size=110).generate(todas_palavras)
# plt.figure(figsize=(10, 7))
# plt.imshow(nuvem_palavras, interpolation='bilinear')
# plt.axis('off')
# plt.show()

'''REMOVENDO PALAVRAS REPETIDAS'''
# nuvem_palavras = WordCloud(width=800, height=500, max_font_size=110, collocations=False).generate(todas_palavras)
# plt.figure(figsize=(10, 7))
# plt.imshow(nuvem_palavras, interpolation='bilinear')
# plt.axis('off')
# plt.show()

'''APLICANDO FUNÇÕES'''
# nuvem_palavras(df, 'avaliacao', 'negativo')
# nuvem_palavras(df, 'avaliacao', 'positivo')

# nltk.download('all')

# frases = ['um produto bom', 'um produto ruim']
# frequencia = nltk.FreqDist(frases)
# print(frequencia.most_common())

'''TOKENIZAÇÃO'''
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

'''REMOVENDO PALAVRAS IRRELEVANTES'''
palavras_irrelevantes = nltk.corpus.stopwords.words('portuguese')

frase_processada = []
for opiniao in df.avaliacao:
    palavras_textos = token_espaco.tokenize(opiniao)
    nova_frase = [palavra for palavra in palavras_textos if palavra not in palavras_irrelevantes]
    frase_processada.append(' '.join(nova_frase))
    
df['tratamento_1'] = frase_processada

# print(df.head())

# classificar_texto(df, 'tratamento_1', 'sentimento')
# Resultado: 81,09%

# grafico_frequencia(df, 'tratamento_1', 20) # Ainda traz pontuação

'''EXEMPLO DE PONTUAÇÃO'''
# frase = 'Esse smartphone superou expectativas, recomendo'
token_pontuacao = tokenize.WordPunctTokenizer()
# token_frase = token_pontuacao.tokenize(frase)
# print(token_frase)

frase_processada = []

for opiniao in df['tratamento_1']:
    palavras_texto = token_pontuacao.tokenize(opiniao)
    nova_frase = [palavra for palavra in palavras_texto if palavra.isalpha() and palavra not in palavras_irrelevantes]
    frase_processada.append(' '.join(nova_frase))
    
df['tratamento_2'] = frase_processada
# print(df.head())

# print(df['tratamento_1'][10]) # Antes do tratamento de pontuação
# print(df['tratamento_2'][10]) # Depois do tratamento

# grafico_frequencia(df, 'tratamento_2', 20)

# Teste unidecode
# frase = 'Um aparelho ótima performance preço bem menor outros aparelhos marcas conhecidas performance semelhante'
# teste = unidecode.unidecode(frase)
# print(teste)

sem_acentos = [unidecode.unidecode(texto) for texto in df['tratamento_2']]
stopwords_sem_acento = [unidecode.unidecode(texto) for texto in palavras_irrelevantes]
df['tratamento_3'] = sem_acentos

frase_processada = []
for opiniao in df['tratamento_3']:
    palavras_texto = token_pontuacao.tokenize(opiniao)
    nova_frase = [palavra for palavra in palavras_texto if palavra not in stopwords_sem_acento]
    frase_processada.append(' '.join(nova_frase))
    
df['tratamento_3'] = frase_processada

# print(df['tratamento_2'][70])
# print(df['tratamento_3'][70])

# grafico_frequencia(df, 'tratamento_3', 20)

# Tratamento de case sensitive
# frase = 'Bom produto otimo custo-beneficio Recomendo Confortavel bem acabado'
# print(frase.lower())

frase_processada = []
for opiniao in df['tratamento_3']:
    opiniao = opiniao.lower()
    palavras_texto = token_pontuacao.tokenize(opiniao)
    nova_frase = [palavra for palavra in palavras_texto if palavra not in stopwords_sem_acento]
    frase_processada.append(' '.join(nova_frase))
    
df['tratamento_4'] = frase_processada

# print(df['tratamento_3'][3])
# print(df['tratamento_4'][3])

# classificar_texto(df, 'tratamento_4', 'sentimento')
# # Acuracia do tratamento 4: 83,75%

stemmer = nltk.RSLPStemmer()
# print(stemmer.stem('gostei')) # Exemplo

frase_processada = []
for opiniao in df['tratamento_4']:
    palavras_texto = token_pontuacao.tokenize(opiniao)
    nova_frase = [stemmer.stem(palavra) for palavra in palavras_texto]
    frase_processada.append(' '.join(nova_frase))
    
df['tratamento_5'] = frase_processada

# print(df['tratamento_4'][3])
# print(df['tratamento_5'][3])

# classificar_texto(df, 'tratamento_5', 'sentimento')
# Acuracia do tratamento 5: 85,11%

'''TF-IDF EXEMPLO'''
# frases = ['Comprei um ótimo produto', 'Comprei um produto péssimo']
tfidf = TfidfVectorizer(lowercase=False, max_features=50)
# matriz = tfidf.fit_transform(frases)
# data = pd.DataFrame(
#     matriz.todense(),
#     columns=tfidf.get_feature_names_out())

# tfidf_bruto = tfidf.fit_transform(df['avaliacao'])

'''TESTANDO COM TF-IDF BRUTO'''
# X_treino, X_teste, y_treino, y_teste = train_test_split(tfidf_bruto, df['sentimento'], random_state=SEED)
# regrassao_logistica.fit(X_treino, y_treino)
# acuracia_tfidf_bruto = regrassao_logistica.score(X_teste, y_teste)
# print(f'Acurácia do modelo com TF-IDF bruto: {acuracia_tfidf_bruto * 100:.2f}%')
# Acurácia do modelo com TF-IDF bruto: 79,54%

'''TESTANDO COM TF-IDF TRATADO'''
tfidf_tratado = tfidf.fit_transform(df['tratamento_5'])

# X_treino, X_teste, y_treino, y_teste = train_test_split(tfidf_tratado, df['sentimento'], random_state=SEED)
# regrassao_logistica.fit(X_treino, y_treino)
# acuracia_tfidf_tratado = regrassao_logistica.score(X_teste, y_teste)
# print(f'Acurácia do modelo com TF-IDF tratado: {acuracia_tfidf_tratado * 100:.2f}%')
# Acurácia do modelo com TF-IDF tratado: 85,14%

'''CAPTURAR CONTEXTO'''
# frase = 'Comprei um produto ótimo'
# frase_separada = token_espaco.tokenize(frase)
# pares = ngrams(frase_separada, 2)
# print(list(pares))

'''TESTANDO COM N-GRAMS'''
# tfidf_50 = TfidfVectorizer(lowercase=False, max_features=50, ngram_range=(1, 2))
# vetor_tfidf = tfidf_50.fit_transform(df['tratamento_5'])
# X_treino, X_teste, y_treino, y_teste = train_test_split(vetor_tfidf, df['sentimento'], random_state=SEED)
# regrassao_logistica.fit(X_treino, y_treino)
# acuracia_tfidf_50 = regrassao_logistica.score(X_teste, y_teste)
# print(f'Acurácia do modelo com TF-IDF 50 e n-grams: {acuracia_tfidf_50 * 100:.2f}%')
# Acurácia do modelo com TF-IDF e n-grams: 85,22%

'''TF-IDF COM N-GRAMS E 100 FEATURES'''
# tfidf_100 = TfidfVectorizer(lowercase=False, max_features=100, ngram_range=(1, 2))
# vetor_tfidf_100 = tfidf_100.fit_transform(df['tratamento_5'])
# X_treino, X_teste, y_treino, y_teste = train_test_split(vetor_tfidf_100, df['sentimento'], random_state=SEED)
# regrassao_logistica.fit(X_treino, y_treino)
# acuracia_tfidf_100 = regrassao_logistica.score(X_teste, y_teste)
# print(f'Acurácia do modelo com TF-IDF 100 e n-grams: {acuracia_tfidf_100 * 100:.2f}%')
# Acurácia do modelo com TF-IDF e n-grams: 88,26%

'''TESTANDO COM N-GRAMS E 1000 FEATURES'''


# Acurácia do modelo com TF-IDF e n-grams: 91,85%