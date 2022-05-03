# IMPORTANDO BIBLIOTECAS
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import string
from unidecode import unidecode
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

import nltk
nltk.download('stopwords')

pontuacoes = string.punctuation

# COLETANDO STOPWORDS
stopwords = nltk.corpus.stopwords.words('portuguese')
stopwords.remove('um')

# IMPORTANDO DADOS
df = pd.read_excel('./data/raw/dados_interacoes_usuarios_identificados.xlsx', dtype='str')
n_extensos = pd.read_csv('./data/raw/numeros_extenso.txt', sep=' - ', names=['NUMERO', 'EXTENSO'])
n_extensos = n_extensos.iloc[:11]


# utils
def substitui_numero_extenso(frase_cliente):

    for i in range(len(n_extensos)):
        if n_extensos['NUMERO'][i] in frase_cliente:
            frase_cliente = frase_cliente.replace(n_extensos['NUMERO'][i], n_extensos['EXTENSO'][i])

    return frase_cliente

#ELIMINANDO FUZZY
def corrige_numeros(frase_cliente, THRESHOLD = 75):

    frase_cliente = frase_cliente.split(' ')

    for i in range(len(frase_cliente)):

        for j in range(len(n_extensos_sem_acento)):

            resultado_fuzz = fuzz.ratio(frase_cliente[i], n_extensos_sem_acento[j])

            if resultado_fuzz >= THRESHOLD:
                frase_cliente[i] = n_extensos_sem_acento[j]

    frase_cliente = ' '.join(frase_cliente)

    return frase_cliente

def elimina_pontos(frase):
    for pontuacao in pontuacoes:
        frase = frase.replace(pontuacao, '')
    return frase


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



# ----------> LIMPEZA DE DADOS <----------

# ELIMINANDO ACENTOS
n_extensos['EXTENSO'] = n_extensos['EXTENSO'].apply(unidecode)
n_extensos_sem_acento = n_extensos['EXTENSO'].tolist()

df['frase_cliente'] = df['frase_cliente'].apply(unidecode)

# ELIMINANDO ZEROS
df['frase_cliente'] = df['frase_cliente'].str.replace('0', '')

# SUBSTITUINDO NÚMEROS POR EXTENSO
df['frase_cliente'] = df['frase_cliente'].apply(substitui_numero_extenso)


# ELIMINANDO PONTUAÇÕES
df['frase_cliente'] = df['frase_cliente'].apply(elimina_pontos)

# ELIMINANDO OPÇÕES INVÁLIDAS (ESCOLHAS SEM NÚMEROS)
for i in range(0, len(df), 2):
    df['frase_cliente'][i+1] = corrige_numeros(df['frase_cliente'][i+1])

    item_escolha = df['frase_cliente'][i+1].split(' ')

    for token in item_escolha:
        if token in n_extensos['EXTENSO'].tolist():
            df['frase_cliente'][i+1] = token

# ELIMINANDO STOPWORDS
df['frase_cliente'] = df['frase_cliente'].apply(eliminando_stopwords)

# ----------> MODELAGEM <----------

lb = LabelEncoder()

df['intent'] = lb.fit_transform(df['intent'])


x = []
y = []

for i in range(0, len(df), 2):
    frase_1 = df['frase_cliente'][i]
    frase_2 = df['frase_cliente'][i+1]

    intencao_2 = df['intent'][i+1]
    x.append(frase_1 + ' ' + frase_2)
    y.append(intencao_2)


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(x)

x_array = X.toarray()

model = GaussianNB()
model.fit(x_array, y)

y_pred = model.predict(x_array)

# RESULTADO DA ACURÁCIA - 100%
accuracy_score(y, y_pred)

# EXPORTANDO PREDIÇÃO   
resultado_predicao = lb.inverse_transform(y_pred)
resultado_predicao = pd.Series(resultado_predicao)
resultado_predicao.to_csv('preenchimento_valores_vazios.csv')

# ----------> INSIGHTS <----------

# FREQUÊNCIA DE PALAVRAS
bag_of_words = pd.DataFrame(x_array)
colunas = list(vectorizer.get_feature_names_out())
bag_of_words.columns = colunas
colunas_sem_numeros = [coluna for coluna in colunas if coluna not in n_extensos['EXTENSO'].tolist()]

bag_of_words_sem_dec_final = bag_of_words[colunas_sem_numeros]
bag_of_words_sem_dec_final_soma = bag_of_words_sem_dec_final.sum()
bag_of_words_sem_dec_final_soma = bag_of_words_sem_dec_final_soma.sort_values(ascending=False)
bag_of_words_sem_dec_final_soma = bag_of_words_sem_dec_final_soma[:9]

plt.bar(bag_of_words_sem_dec_final_soma.index, bag_of_words_sem_dec_final_soma)
plt.title('Frequência de Palavras')
plt.show()

cont_atendeu = 0
cont_nao_atendeu = 0

for i in range(0, len(df), 2):
    if df['frase_cliente'][i+1] in n_extensos['EXTENSO'].tolist():
        cont_atendeu+=1
    else:
        cont_nao_atendeu+=1

plt.bar(['Opções apropriadas', 'Opções não apropriadas'], [cont_atendeu, cont_nao_atendeu])
plt.title('Proporção de opções apropriadas')
plt.show()