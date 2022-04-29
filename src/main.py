import pandas as pd
from sklearn.preprocessing import LabelEncoder


df = pd.read_excel('./data/dados_interacoes_usuarios_identificados.xlsx')

df_test = pd.read_excel('./data/dados_interacoes_usuarios_identificados.xlsx')



n_extensos = pd.read_csv('./data/numeros_extenso.txt', sep=' - ', names=['NUMERO', 'EXTENSO'])


n_extensos = n_extensos.iloc[:11]


def substitui_extenso(frase):
    



lb = LabelEncoder()

df_test['resposta_assistente_virtual'] = lb.fit_transform(df_test['resposta_assistente_virtual'])


df_test.iloc[0]
