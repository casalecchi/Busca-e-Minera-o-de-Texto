import logging
import pandas as pd
import generate_inverted_list as gil
import numpy as np
from datetime import datetime


def read_config_file(config_file):
    """"Retorna os paths dos arquivos que contêm o modelo, as consultas e também o arquivo que será escrito como resultado. 
    Lê o arquivo de configuração do módulo."""

    logging.basicConfig(filename='../results/busca.log', filemode='a',format='%(asctime)s - %(message)s', level=logging.INFO, force=True)
    logging.info("Started reading the configuration file.")
    model_file = "../results/"
    queries_file = "../results/"
    results_file = "../results/"
    cfg_path = "../data/" + config_file
    stemmer = False

    with open(cfg_path, "r") as config_file:
        for line in config_file.readlines():
            instruction, filename = line.split("=")
            filename = filename.strip()
            
            if instruction == "MODELO":
                model_file += filename
            elif instruction == "CONSULTAS":
                queries_file += filename
            elif instruction == "RESULTADOS":
                results_file += filename

    if results_file.find("-stemmer") != -1:
        stemmer = True
        logging.info("Stemming option is ON.")
    else:
        logging.info("Stemming option is OFF.")
            
    logging.info("Finished reading the configuration file.")
    
    return model_file, queries_file, results_file, stemmer


def get_model(model_file):
    """Retorna o modelo como DataFrame. É necessário passar o path do arquivo do modelo."""
    
    logging.basicConfig(filename='../results/busca.log', filemode='a',format='%(asctime)s - %(message)s', level=logging.INFO, force=True)
    logging.info("Loading model.")
    model = pd.read_csv(model_file, sep=";")
    model.set_index(["Token"], inplace=True)
    logging.info("Model loaded.")
    
    return model


def get_queries(queries_file, stemmer):
    """Retorna o arquivo de consultas como uma lista invertida em um DataFrame. O texto dessas consultas são pré-processados da mesma maneira 
    que o texto dos documentos. É necessário passar o path desse arquivo"""

    logging.basicConfig(filename='../results/busca.log', filemode='a',format='%(asctime)s - %(message)s', level=logging.INFO, force=True)
    logging.info("Loading queries.")
    
    queries = pd.read_csv(queries_file, sep=";")
    queries.set_index(["QueryNumber"], inplace=True)

    for number, text in queries.itertuples():
        processed_text = gil.preproccess_text(text, stemmer)
        queries.at[number, "QueryText"] = processed_text

    logging.info("Queries loaded.")
    
    return queries


def insert_queries(model, queries):
    """Retorna o modelo concatenado com as consultas pré-processadas."""

    logging.basicConfig(filename='../results/busca.log', filemode='a',format='%(asctime)s - %(message)s', level=logging.INFO, force=True)
    logging.info("Inserting queries in model.")
    start_time = datetime.now()

    shape = (model.shape[0], 1)

    # Como o df queries, será como uma lista invertida, adicionamos as QueryNumber como documentos no modelo
    # e os termos presentes em cada Query terá o peso 1.0, como especificado no enunciado do trabalho
    for qnumber, qtext in queries.itertuples():
        # Primeiro criamos a coluna zerada da Query e a concatenamos no modelo
        zeros = pd.DataFrame(np.zeros(shape), index=model.index, columns=[f"Q{qnumber}"])
        model = pd.concat([model, zeros], axis=1)

        for word in qtext:
            # Caso uma palavra não esteja no modelo, a adicionamos
            if not word in model.index:
                zeros = pd.DataFrame(np.zeros((1, len(model.columns))), index=[word], columns=model.columns)
                model = pd.concat([model, zeros], axis=0)
                shape = (model.shape[0], 1)

            # Adicionamos o peso da palavra na coluna da query correspondente
            model.at[word, f"Q{qnumber}"] += 1

    time_taken = datetime.now() - start_time
    logging.info(f"Queries inserted in model. Time taken: {time_taken}")

    return model


def get_vector(document, model):
    """Retorna o vetor de um documento no modelo."""
    return model[str(document)].to_numpy()  


def get_vector_size(vector):
    """Retorna a norma de um vetor."""

    return np.linalg.norm(vector)


def sim_cos(query, document, model):
    """Retorna a similaridade de cossenos entre uma consulta e um documento, dado um modelo em que ambos estão presentes."""

    q = get_vector(query, model)
    d = get_vector(document, model)

    q_dot_d = np.dot(q, d)
    qxd = get_vector_size(q) * get_vector_size(d)
    
    return q_dot_d / qxd


def get_ranking(model, queries):
    """Retorna o ranking no formato de um DataFrame. Ele contém as consultas como colunas e os documentos em suas linhas.
    O valor representa o valor calculado da similaridade de cossenos calculados entre dada consulta e determinado documento."""

    logging.basicConfig(filename='../results/busca.log', filemode='a',format='%(asctime)s - %(message)s', level=logging.INFO, force=True)
    logging.info("Started getting rankings of each query by each document.")
    start_time = datetime.now()
    
    # Temos e modelo e adicionamos as consultas
    model = insert_queries(model, queries)
    ranking = pd.DataFrame()
    shape = (model.shape[1] - 99, 1)

    for query in queries.index:
        q = f"Q{query}"
        zeros = pd.DataFrame(np.zeros(shape), index=model.columns[:-99], columns=[q])
        ranking = pd.concat([ranking, zeros], axis=1)
        
        # Iterar pelas colunas de documento do modelo, retirando as de query
        for document in model.columns[:-99]:
            result = sim_cos(q, document, model)
            ranking.loc[document, q] = result

    time_taken = datetime.now() - start_time
    logging.info(f"Rankings created. Time taken: {time_taken}")
    
    return ranking


def get_results(file, ranking):
    """É gerado o CSV com os resultados, passando o path desse arquivo e o ranking."""

    logging.basicConfig(filename='../results/busca.log', filemode='a',format='%(asctime)s - %(message)s', level=logging.INFO, force=True)
    logging.info("Generating the results file. CSV format file with 'QueryNumber; [position in ranking, doc_number, value of sim_cos]'.")

    with open(file, 'w') as results:
        results.write("QueryNumber;DocInfos\n")
        for query in ranking.columns:
            query_number = query.replace('Q', '')
            sorted_ranking = ranking[query].sort_values(ascending=False)
            position_ranking = 1
            for doc_number, cos in sorted_ranking.items():
                # se a similaridade de cossenos entre um documento e uma consulta for igual a zero,
                # o documento não entra no ranking da consulta
                if cos == 0:
                    break
                doc_infos = [position_ranking, doc_number, cos]
                position_ranking += 1
                results.write(f"{query_number};{doc_infos}\n")
    
    logging.info("Results file created.")


def start_exec():
    logging.basicConfig(filename='../results/busca.log', filemode='a',format='%(asctime)s - %(message)s', level=logging.INFO, force=True)
    logging.info("Module searcher started.")


def finish_exec():
    logging.basicConfig(filename='../results/busca.log', filemode='a',format='%(asctime)s - %(message)s', level=logging.INFO, force=True)
    logging.info("Module searcher finished execution.")


# Configuramos o arquivo que será o log do módulo
logging.basicConfig(filename='../results/busca.log', filemode='w',format='%(asctime)s - %(message)s', level=logging.INFO, force=True)
logging.info("Log created.")