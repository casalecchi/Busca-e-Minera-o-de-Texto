import logging
import pandas as pd
import generate_inverted_list as gil
import numpy as np
from datetime import datetime


def read_config_file(config_file):
    logging.info("Started reading the configuration file.")
    model_file = "../results/"
    queries_file = "../results/"
    results_file = "../results/"
    cfg_path = "../data/" + config_file

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
            
    logging.info("Finished reading the configuration file.")
    return model_file, queries_file, results_file


def get_model(model_file):
    logging.info("Loading model.")

    model = pd.read_csv(model_file, sep=";")
    model.set_index(["Token"], inplace=True)

    logging.info("Model loaded.")
    return model


def get_queries(queries_file):
    logging.info("Loading queries.")

    queries = pd.read_csv(queries_file, sep=";")
    queries.set_index(["QueryNumber"], inplace=True)
    for number, text in queries.itertuples():
        processed_text = gil.preproccess_text(text)
        queries.at[number, "QueryText"] = processed_text

    logging.info("Queries loaded.")
    return queries


def insert_queries(model, queries):
    logging.info("Inserting queries in model.")
    start_time = datetime.now()

    shape = (model.shape[0], 1)
    for qnumber, qtext in queries.itertuples():
        zeros = pd.DataFrame(np.zeros(shape), index=model.index, columns=[f"Q{qnumber}"])
        model = pd.concat([model, zeros], axis=1)
        for word in qtext:
            if not word in model.index:
                zeros = pd.DataFrame(np.zeros((1, len(model.columns))), index=[word], columns=model.columns)
                model = pd.concat([model, zeros], axis=0)
                shape = (model.shape[0], 1)
            model.at[word, f"Q{qnumber}"] += 1

    time_taken = datetime.now() - start_time
    logging.info(f"Queries inserted in model. Time taken: {time_taken}")
    return model


def get_vector(document, model):
    return model[str(document)].to_numpy()  


def get_vector_size(vector):
    return np.linalg.norm(vector)


def sim_cos(query, document, model):
    q = get_vector(query, model)
    d = get_vector(document, model)

    q_dot_d = np.dot(q, d)
    qxd = get_vector_size(q) * get_vector_size(d)
    
    return q_dot_d / qxd


def get_ranking(model, queries):
    logging.info("Started getting rankings of each query by each document.")
    start_time = datetime.now()

    model = insert_queries(model, queries)
    ranking = pd.DataFrame()
    shape = (model.shape[1] - 99, 1)

    for query in queries.index:
        q = f"Q{query}"
        zeros = pd.DataFrame(np.zeros(shape), index=model.columns[:-99], columns=[q])
        ranking = pd.concat([ranking, zeros], axis=1)
        # Iterar pelas colunas de documento, retirando as de query
        for document in model.columns[:-99]:
            result = sim_cos(q, document, model)
            ranking.loc[document, q] = result

    time_taken = datetime.now() - start_time
    logging.info(f"Rankings created. Time taken: {time_taken}")
    return ranking


def get_results(file, ranking):
    logging.info("Generating the results file. CSV format file with 'QueryNumber; [position in ranking, doc_number, value of sim_cos]'.")
    with open(file, 'w') as results:
        results.write("QueryNumber;DocInfos\n")
        for query in ranking.columns:
            query_number = query.replace('Q', '')
            sorted_ranking = ranking[query].sort_values(ascending=False)
            position_ranking = 1
            for doc_number, cos in sorted_ranking.items():
                if cos == 0:
                    break
                doc_infos = [position_ranking, doc_number, cos]
                position_ranking += 1
                results.write(f"{query_number};{doc_infos}\n")
    logging.info("Results file created.")


logging.basicConfig(filename='../results/busca.log', filemode='w',format='%(asctime)s - %(message)s', level=logging.INFO, force=True)
logging.info("Log created.")
model_file, queries_file, results_file = read_config_file("busca.cfg")
model = get_model(model_file)
queries = get_queries(queries_file)
ranking = get_ranking(model, queries)
get_results(results_file, ranking)

logging.info("Finished execution.")