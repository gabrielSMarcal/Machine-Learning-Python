import joblib
import matplotlib.pyplot as plt

tfidf               = joblib.load('tfidf_vectorizer.pkl')
regressao_logistica = joblib.load('modelo_regressao_logistica.pkl')

