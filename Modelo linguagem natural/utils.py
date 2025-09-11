import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

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
