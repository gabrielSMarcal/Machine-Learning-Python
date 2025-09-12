import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('dataset_avaliacoes.csv')

# print(df.value_counts('sentimento'))
# print('positiva \n')
# print(df.avaliacao[2])

text = ['Comprei um produto ótimo', 'Comprei um produto ruim']

vetorizar = CountVectorizer()
bag_of_words = vetorizar.fit_transform(text)
# print(bag_of_words)

matriz_esparsa = pd.DataFrame.sparse.from_spmatrix(bag_of_words, columns=vetorizar.get_feature_names_out())
# print(matriz_esparsa)

# vetorizar = CountVectorizer(lowercase=False)
# bag_of_words = vetorizar.fit_transform(df.avaliacao)
# print(bag_of_words.shape)

vetorizar = CountVectorizer(lowercase=False, max_features=50)
bag_of_words = vetorizar.fit_transform(df.avaliacao)
# print(bag_of_words.shape)

matriz_esparsa_avaliacoes = pd.DataFrame.sparse.from_spmatrix(bag_of_words, columns=vetorizar.get_feature_names_out())
# print(matriz_esparsa_avaliacoes)

X_treino, X_teste, y_treino, y_teste = train_test_split(bag_of_words, df.sentimento, test_size=0.2,
                                                        random_state=4978)

regrassao_logistica = LogisticRegression()
regrassao_logistica.fit(X_treino, y_treino)
# acuracia = regrassao_logistica.score(X_teste, y_teste)
# print(f'A acurácia do modelo foi de {acuracia * 100:.2f}%')