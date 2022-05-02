import pandas as pd
from sklearn.preprocessing import LabelEncoder
import string
from unidecode import unidecode
from fuzzywuzzy import fuzz


df = pd.read_excel('./data/dados_interacoes_usuarios_identificados.xlsx')
df_test = pd.read_excel('./data/dados_interacoes_usuarios_identificados.xlsx', dtype='str')

n_extensos = pd.read_csv('./data/numeros_extenso.txt', sep=' - ', names=['NUMERO', 'EXTENSO'])


n_extensos = n_extensos.iloc[:11]


def substitui_numero_extenso(frase_cliente):

    for i in range(len(n_extensos)):
        if n_extensos['NUMERO'][i] in frase_cliente:
            frase_cliente = frase_cliente.replace(n_extensos['NUMERO'][i], n_extensos['EXTENSO'][i])

    return frase_cliente


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




df_test['frase_cliente'] = df_test['frase_cliente'].str.replace('0', '')





num_teste = n_extensos['EXTENSO'].apply(unidecode)
num_teste = num_teste.tolist()





#ELIMINANDO FUZZY
def corrige_numeros(frase_cliente, THRESHOLD = 70):

    frase_cliente = frase_cliente.split(' ')

    for i in range(len(frase_cliente)):

        for j in range(len(num_teste)):

            resultado_fuzz = fuzz.ratio(frase_cliente[i], num_teste[j])

            if resultado_fuzz > THRESHOLD:
                frase_cliente[i] = num_teste[j]

    frase_cliente = ' '.join(frase_cliente)

    return frase_cliente




df_test['frase_cliente'].apply(corrige_numeros)




df_test['frase_cliente'][27]




#--------------------

lb = LabelEncoder()

df_test['resposta_assistente_virtual'] = lb.fit_transform(df_test['resposta_assistente_virtual'])

