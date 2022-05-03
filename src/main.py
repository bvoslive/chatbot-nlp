import pandas as pd
from sklearn.preprocessing import LabelEncoder
import string
from unidecode import unidecode
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

import nltk
nltk.download('stopwords')

stopwords = nltk.corpus.stopwords.words('portuguese')




df = pd.read_excel('./data/dados_interacoes_usuarios_identificados.xlsx')

df_test = pd.read_excel('./data/dados_interacoes_usuarios_identificados.xlsx', dtype='str')
n_extensos = pd.read_csv('./data/numeros_extenso.txt', sep=' - ', names=['NUMERO', 'EXTENSO'])

n_extensos = n_extensos.iloc[:11]

n_extensos['EXTENSO'] = n_extensos['EXTENSO'].apply(unidecode)

df_test['frase_cliente'] = df_test['frase_cliente'].apply(unidecode)


def substitui_numero_extenso(frase_cliente):

    for i in range(len(n_extensos)):
        if n_extensos['NUMERO'][i] in frase_cliente:
            frase_cliente = frase_cliente.replace(n_extensos['NUMERO'][i], n_extensos['EXTENSO'][i])

    return frase_cliente


df_test['frase_cliente'] = df_test['frase_cliente'].str.replace('0', '')


df_test['frase_cliente'] = df_test['frase_cliente'].apply(substitui_numero_extenso)


pontuacoes = string.punctuation

def elimina_pontos(frase):
    for pontuacao in pontuacoes:
        frase = frase.replace(pontuacao, '')
    return frase

df_test['frase_cliente'] = df_test['frase_cliente'].apply(elimina_pontos)

# ELIMINANDO ACENTOS
df_test['frase_cliente'] = df_test['frase_cliente'].apply(unidecode)

#frases_cliente = df_test['frase_cliente'].str.split()




num_teste = n_extensos['EXTENSO'].apply(unidecode)




num_teste = num_teste.tolist()






#ELIMINANDO FUZZY
def corrige_numeros(frase_cliente, THRESHOLD = 75):

    frase_cliente = frase_cliente.split(' ')

    for i in range(len(frase_cliente)):

        for j in range(len(num_teste)):

            resultado_fuzz = fuzz.ratio(frase_cliente[i], num_teste[j])

            if resultado_fuzz >= THRESHOLD:
                frase_cliente[i] = num_teste[j]

    frase_cliente = ' '.join(frase_cliente)

    return frase_cliente



df_test['frase_cliente'] = df_test['frase_cliente'].apply(corrige_numeros)







def eliminando_stopwords(frase):

    tokens = frase.split(' ')
    for k in range(len(tokens)):
        for stopword in stopwords:
            if tokens[k] == stopword:
                tokens[k] = 'ELIMINA'


    frase = ' '.join(tokens)
    frase = frase.replace(' ELIMINA', '')
    frase = frase.replace('ELIMINA ', '')
    frase = frase.replace('ELIMINA', '')

    return frase

df_test['frase_cliente'] = df_test['frase_cliente'].apply(eliminando_stopwords)



#--------------------

lb = LabelEncoder()
lb2 = LabelEncoder()

df_test['resposta_assistente_virtual'] = lb.fit_transform(df_test['resposta_assistente_virtual'])
df_test['intent'] = lb2.fit_transform(df_test['intent'])







x = []
y = []

for i in range(0, len(df_test), 2):
    frase_1 = df_test['frase_cliente'][i]
    frase_2 = df_test['frase_cliente'][i+1]


    frase_2_splited = frase_2.split()




    for j in range(len(frase_2_splited)):

        if frase_2_splited[j] in n_extensos['EXTENSO'].tolist():
            



    intencao_2 = df_test['intent'][i+1]

    x.append(frase_1 + ' ' + frase_2)
    y.append(intencao_2)



y





vectorizer = CountVectorizer()
X = vectorizer.fit_transform(x)


from sklearn.naive_bayes import GaussianNB


nb_tfidf = GaussianNB()
nb_tfidf.fit(X, y)

y_pred = nb_tfidf.predict(X)


pd.DataFrame({'y':y, 'pred':y_pred})





