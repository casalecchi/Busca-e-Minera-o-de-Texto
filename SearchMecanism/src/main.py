import processing_queries as pq
import generate_inverted_list as gil
import indexer
import searcher
import logging
import sys


logging.basicConfig(filename='../results/pc.log', filemode='w',format='%(asctime)s - %(message)s', level=logging.INFO)
logging.info("Log created.")
read, queries, expected = pq.read_config("pc.cfg")
xml_root = pq.get_xml_root(read)
pq.get_queries_file(queries, xml_root)
pq.get_expected_file(expected, xml_root)
logging.info("Finished execution.")

logging.basicConfig(filename='../results/gli.log', filemode='w',format='%(asctime)s - %(message)s', level=logging.INFO)
logging.info("Log created.")
logging.info("Downloading 'punkt' and 'stopwords' from nltk-data...")
logging.info("Finished downloading 'punkt' and 'stopwords'.")
read_files, write_file = gil.read_config_file("gli.cfg")
gil.get_tokens_file(read_files, write_file)
logging.info("Finished execution.")

logging.basicConfig(filename='../results/index.log', filemode='w',format='%(asctime)s - %(message)s', level=logging.INFO)
logging.info("Log created.")
type_tf = sys.argv[0]
tokens, model = indexer.read_config_file("index.cfg")
indexer.save_model(model, tokens, type_tf)
logging.info("Finished execution.")

logging.basicConfig(filename='../results/busca.log', filemode='w',format='%(asctime)s - %(message)s', level=logging.INFO, force=True)
logging.info("Log created.")
model_file, queries_file, results_file = searcher.read_config_file("busca.cfg")
model = searcher.get_model(model_file)
queries = searcher.get_queries(queries_file)
ranking = searcher.get_ranking(model, queries)
searcher.get_results(results_file, ranking)
searcher.logging.info("Finished execution.")