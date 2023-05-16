from datetime import datetime
from tqdm import tqdm
import sys
import logging
import pandas as pd
import numpy as np
from math import log10


def read_config_file(config_file):
    """"Retorna os paths dos arquivos de leitura e de escrita. Lê o arquivo de configuração do módulo."""

    logging.basicConfig(filename='../results/index.log', filemode='a',format='%(asctime)s - %(message)s', level=logging.INFO, force=True)
    logging.info("Started reading the configuration file.")
    read_file = "../results/"
    write_file = "../results/"
    cfg_path = "../data/" + config_file

    with open(cfg_path, "r") as config_file:
        for line in config_file.readlines():
            instruction, filename = line.split("=")
            filename = filename.strip()
            
            if instruction == "LEIA":
                read_file += filename
            elif instruction == "ESCREVA":
                write_file += filename

    logging.info("Finished reading the configuration file.")
    return (read_file, write_file)


def get_term_document_matrix(tokens_file):
    """Retornamos um DataFrame do pandas com a matriz termo documento. Os índices são os termos e as colunas são os números dos documentos.
    O valor correspondente de termo x documento representa a quantidade de vezes que uma palavra aparece em determinado documento. É preciso
    passar como argumento o nome do arqiuvo que contém a lista invertida."""

    logging.basicConfig(filename='../results/index.log', filemode='a',format='%(asctime)s - %(message)s', level=logging.INFO, force=True)
    logging.info("Started generating the term document matrix.")
    start_time = datetime.now()
    number_tokens = 0
    path = "../results/" + tokens_file

    # Lemos e carregamos a lista invertida como um DataFrame para usar de "base" para matriz
    inverted_list = pd.read_csv(path, sep=';', converters={"Appearance": pd.eval})

    # São carregadas apenas as palavras na matriz e a colocadas como índices
    matrix = pd.DataFrame(inverted_list["Token"])
    matrix.set_index(["Token"], inplace=True)

    # Shape com a quantidade de palavras
    shape = (matrix.shape[0], 1)

    # Iteração com a lista invertida
    # Teremos a palavra e a lista de documentos em que ela está presente
    for token, docs in inverted_list.itertuples(index=False):
        number_tokens += 1

        # Para cada documento na lista
        for doc in docs:
            # caso esse documento já exista na matriz, apenas adicionamos uma unidade no termo correspondente
            if str(doc) in matrix.columns:
                matrix.at[token, str(doc)] += 1

            # caso o documento ainda não esteja na matriz, iremos criar sua coluna contendo apenas zero, concatenar na matriz
            # e adiconar o termo atual
            else:
                zeros = pd.DataFrame(np.zeros(shape), index=inverted_list["Token"], columns=[str(doc)])
                matrix = pd.concat([matrix, zeros], axis=1)
                matrix.at[token, str(doc)] = 1
    
    time_taken = datetime.now() - start_time
    logging.info(f"Finished generating the term document matrix. Time taken: {time_taken}.")
    logging.info(f"Matrix has {len(matrix.index)} tokens and {len(matrix.columns)} documents.")
    
    return matrix


def get_n(matrix):
    """Retorna o número total de documentos da matriz."""
    
    N = len(matrix.columns)
    return N


def get_nj(token, matrix):
    """Retorna o número total de documentos da matriz que contém um dado termo."""

    row = matrix.loc[token]
    return row.astype(bool).sum()  # Tudo que tiver algum valor diferente de zero será 1, 
                                   # e então somamos todas as colunas


def get_tf(token, document, matrix):
    """Retorna o número total em que um dado termo aparece em um determinado documento (frequência bruta)."""

    return int(matrix.loc[token, str(document)])


def get_tfn(token, document, matrix):
    """Retorna o valor da frequência normalizada de um dado termo em um determiando documento. Esse valor é obtido dividindo
    a frequência bruta desse termo, pela frequência bruta do termo que mais aparece no documento."""

    tf = get_tf(token, document, matrix)
    biggest_tf = int(matrix.loc[:, str(document)].max())
    return tf / biggest_tf


def get_idf(token, matrix):
    """Retorna a fórmula de idf já calculada."""

    return log10(get_n(matrix) / get_nj(token, matrix))


def get_model(matrix, type_tf="tf"):
    """Retorna o DataFrame que é construído como modelo. O formato desse modelo pode ser visto no arquivo 'modelo.txt' no diretório
    'SearchMechanism'. Basicamente ele é construído como a matriz de termo e documento, mas em vez de ter a frequência bruta em seus
    valores, ele possui os pesos calculados pelo tf-idf. É necessário passar a matriz em formato de DataFrame e o tipo do tf que será
    utilizado, sendo 'tf' ou 'tfn'."""

    logging.basicConfig(filename='../results/index.log', filemode='a',format='%(asctime)s - %(message)s', level=logging.INFO, force=True)
    # Mensagem usada no log do módulo
    msg = "Started generating the model with tf"
    if type_tf == "tf":
        msg += "."
    else:
        msg += " normalized."
    msg += " The model usually takes 4~7 minutes to be created."
    logging.info(msg)
    
    start_time = datetime.now()
    weights = matrix.copy()

    for token in tqdm(weights.index):
        # para cada palavra, primeiro calculamos seu idf
        idf = get_idf(token, weights)

        for document in weights.columns:
            # Para cada documento e palavra, calculamos o seu tf (ou tfn)
            tf = get_tfn(token, document, matrix) if type_tf == "tfn" else get_tfn(token, document, matrix) 
            # Calculamos o peso
            wij = tf * idf
            
            # Adicionamos o peso no modelo
            weights.loc[token, str(document)] = wij

    time_taken = datetime.now() - start_time
    logging.info(f"Finished generating the model. Time taken: {time_taken}.")
    
    return weights

def save_matrix(save_path, tokens_file):
    """Primeiro geramos a matriz e depois salvamos a mesma em um path passado. É necessário passar o arquivo CSV contendo a lista invertida.
    Ao final é retornada a matriz."""

    logging.basicConfig(filename='../results/index.log', filemode='a',format='%(asctime)s - %(message)s', level=logging.INFO, force=True)

    matrix = get_term_document_matrix(tokens_file)
    matrix.to_csv(save_path, sep=";")
    logging.info("Term document matrix saved.")
    
    return matrix


def save_model(path, tokens_file, type_tf):
    """Salvamos o modelo como um arquivo CSV seguindo o path passado como argumento. É necessário passar a lista invertida e o tipo do
    tf que será utilizado (tf | tfn)."""

    logging.basicConfig(filename='../results/index.log', filemode='a',format='%(asctime)s - %(message)s', level=logging.INFO, force=True)
    
    matrix = save_matrix("../results/matrix.csv", tokens_file)
    model = get_model(matrix, type_tf)
    model.to_csv(path, sep=";")
    logging.info("Model saved.")


def start_exec():
    logging.basicConfig(filename='../results/index.log', filemode='a',format='%(asctime)s - %(message)s', level=logging.INFO, force=True)
    logging.info("Module indexer started.")


def finish_exec():
    logging.basicConfig(filename='../results/index.log', filemode='a',format='%(asctime)s - %(message)s', level=logging.INFO, force=True)
    logging.info("Module indexer finished execution.")
    

# Configuramos o arquivo que será o log do módulo
logging.basicConfig(filename='../results/index.log', filemode='w',format='%(asctime)s - %(message)s', level=logging.INFO, force=True)
logging.info("Log created.")