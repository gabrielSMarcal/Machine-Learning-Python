import pandas as pd

df = pd.read_csv('dataset_avaliacoes.csv')

# print(df.value_counts('sentimento'))
print('positiva \n')
print(df.avaliacao[2])