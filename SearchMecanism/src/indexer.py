from datetime import datetime
from tqdm import tqdm
import sys
import logging
import pandas as pd
import numpy as np
from math import log10


def read_config_file(config_file):
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
    logging.info("Started generating the term document matrix.")
    start_time = datetime.now()
    number_tokens = 0
    path = "../results/" + tokens_file

    inverted_list = pd.read_csv(path, sep=';', converters={"Appearance": pd.eval})
    matrix = pd.DataFrame(inverted_list["Token"])
    matrix.set_index(["Token"], inplace=True)
    shape = (matrix.shape[0], 1)

    for token, docs in inverted_list.itertuples(index=False):
        number_tokens += 1
        for doc in docs:
            if str(doc) in matrix.columns:
                matrix.at[token, str(doc)] += 1
            else:
                zeros = pd.DataFrame(np.zeros(shape), index=inverted_list["Token"], columns=[str(doc)])
                matrix = pd.concat([matrix, zeros], axis=1)
                matrix.at[token, str(doc)] = 1
    
    time_taken = datetime.now() - start_time
    logging.info(f"Finished generating the term document matrix. Time taken: {time_taken}.")
    logging.info(f"Matrix has {len(matrix.index)} tokens and {len(matrix.columns)} documents.")
    return matrix


def get_n(matrix):
    N = len(matrix.columns)
    return N


def get_nj(token, matrix):
    row = matrix.loc[token]
    return row.astype(bool).sum()  # Tudo que tiver algum valor diferente de zero será 1, 
                                   # e então somamos todas as colunas


def get_tf(token, document, matrix):
    return int(matrix.loc[token, str(document)])


def get_tfn(token, document, matrix):
    tf = get_tf(token, document, matrix)
    biggest_tf = int(matrix.loc[:, str(document)].max())
    return tf / biggest_tf


def get_idf(token, matrix):
    return log10(get_n(matrix) / get_nj(token, matrix))


def get_model(matrix, type_tf="tf"):
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
        idf = get_idf(token, weights)
        for document in weights.columns:
            tf = get_tfn(token, document, matrix) if type_tf == "tfn" else get_tfn(token, document, matrix) 
            wij = tf * idf
            weights.loc[token, str(document)] = wij

    time_taken = datetime.now() - start_time
    logging.info(f"Finished generating the model. Time taken: {time_taken}.")
    return weights

def save_matrix(save_path, tokens_file):
    matrix = get_term_document_matrix(tokens_file)
    matrix.to_csv(save_path, sep=";")
    logging.info("Term document matrix saved.")
    return matrix


def save_model(path, tokens_file, type_tf):
    matrix = save_matrix("../results/matrix.csv", tokens_file)
    model = get_model(matrix, type_tf)
    model.to_csv(path, sep=";")
    logging.info("Model saved.")
