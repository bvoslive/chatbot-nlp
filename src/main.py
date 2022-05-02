import pandas as pd
from sklearn.preprocessing import LabelEncoder
import string
from unidecode import unidecode

df = pd.read_excel('./data/dados_interacoes_usuarios_identificados.xlsx')
df_test = pd.read_excel('./data/dados_interacoes_usuarios_identificados.xlsx', dtype='str')

n_extensos = pd.read_csv('./data/numeros_extenso.txt', sep=' - ', names=['NUMERO', 'EXTENSO'])


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


df_test['frase_cliente'].apply(elimina_pontos)

elimina_pontos = lambda x: [x.replace(pontuacao, '') for pontuacao in pontuacoes]






lb = LabelEncoder()

df_test['resposta_assistente_virtual'] = lb.fit_transform(df_test['resposta_assistente_virtual'])

