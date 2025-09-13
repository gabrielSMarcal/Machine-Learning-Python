import joblib
import nltk
from nltk import tokenize
from unidecode import unidecode

tfidf               = joblib.load('tfidf_vectorizer.pkl')
regressao_logistica = joblib.load('modelo_regressao_logistica.pkl')

palavras_irrelevantes = nltk.corpus.stopwords.words('portuguese')
token_espaco = tokenize.WordPunctTokenizer()
stemmer = nltk.RSLPStemmer()

def processar_avaliacao(avaliacao):
    
    # Passo 1: Tokenização
    tokens = token_espaco.tokenize(avaliacao)

    # Passo 2: Remoção de palavras irrelevantes
    frase_processada = [palavra for palavra in tokens if palavra.lower() not in palavras_irrelevantes]
    
    # Passo 3: Remover pontuação
    frase_processada = [palavra for palavra in frase_processada if palavra.isalpha()]
    
    # Passo 4: Remover acentuação
    frase_processada = [unidecode.unidecode(palavra) for palavra in frase_processada]
    
    # Passo 5: Stemming
    frase_processada = [stemmer.stem(palavra) for palavra in frase_processada]
    
    return ' '.join(frase_processada)

