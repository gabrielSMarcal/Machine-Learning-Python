import joblib
import nltk
import pandas as pd
import unidecode
from nltk import tokenize

tfidf               = joblib.load('tfidf_vectorizer.pkl')
regressao_logistica = joblib.load('modelo_regressao_logistica.pkl')

palavras_irrelevantes = nltk.corpus.stopwords.words('portuguese')
token_pontuacao = tokenize.WordPunctTokenizer()
stemmer = nltk.RSLPStemmer()

def processar_avaliacao(avaliacao):
    
    # Passo 1: Tokenização
    tokens = token_pontuacao.tokenize(avaliacao)

    # Passo 2: Remoção de palavras irrelevantes
    frase_processada = [palavra for palavra in tokens if palavra.lower() not in palavras_irrelevantes]
    
    # Passo 3: Remover pontuação
    frase_processada = [palavra for palavra in frase_processada if palavra.isalpha()]
    
    # Passo 4: Remover acentuação
    frase_processada = [unidecode.unidecode(palavra) for palavra in frase_processada]
    
    # Passo 5: Stemming para retornar apenas radicais
    frase_processada = [stemmer.stem(palavra) for palavra in frase_processada]
    
    return ' '.join(frase_processada)

'''NOVAS AVALIAÇÕES PARA PREVER'''
novas_avaliacoes = ["Ótimo produto, super recomendo!",
                 "A entrega atrasou muito! Estou decepcionado com a compra",
                 "Muito satisfeito com a compra. Além de ter atendido as expectativas, o preço foi ótimo",
                 "Horrível!!! O produto chegou danificado e agora estou tentando fazer a devolução.",
                 '''Rastreando o pacote, achei que não fosse recebê-lo, pois, na data prevista, estava sendo entregue em outra cidade.
                 Mas, no fim, deu tudo certo e recebi o produto.Produto de ótima qualidade, atendendo bem as minhas necessidades e por
                 um preço super em conta.Recomendo.''']

novas_avaliacoes_processadas = [processar_avaliacao(avaliacao) for avaliacao in novas_avaliacoes]

# print(novas_avaliacoes_processadas)

novas_avaliacoes_tfidf = tfidf.transform(novas_avaliacoes_processadas)
precicoes = regressao_logistica.predict(novas_avaliacoes_tfidf)

df_previsoes = pd.DataFrame({
    'Avaliação': novas_avaliacoes,
    'Sentimento previsto': precicoes
})

'''RESULTADO FINAL'''
print(df_previsoes)