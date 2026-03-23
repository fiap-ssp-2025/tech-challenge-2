# usar o pandas para carregar o dataset e mostrar o nome das colunas
import pandas as pd

# carregar o primeiro dataset da pas data/raw no formato .parquet
df = pd.read_parquet('data/raw/VIOLBR24.parquet')

# mostrar o nome de todas as colunas do dataset com to.list()
# porque o df.columns exibe apenas um resumo das colunas
#print(df.columns.to_list())

#mostrar as cinco primeiras linhas dos campos I

# Verificar se todos os arquivos .parquet possuem a mesma estrutura de colunas
import os
# Listar todos os arquivos .parquet na pasta data/raw
parquet_files = [f for f in os.listdir('data/raw') if f.endswith('.parquet')]
# Verificar as colunas de cada arquivo
for file in parquet_files:
    df_temp = pd.read_parquet(f'data/raw/{file}')
    print(f'Colunas do arquivo {file}: {df_temp.columns.to_list()}')
# Se todos os arquivos tiverem a mesma estrutura de colunas, podemos concatená-los em um único DataFrame
# Concatenar todos os arquivos .parquet em um único DataFrame
df_list = [pd.read_parquet(f'data/raw/{file}') for file in parquet_files]
df = pd.concat(df_list, ignore_index=True)

