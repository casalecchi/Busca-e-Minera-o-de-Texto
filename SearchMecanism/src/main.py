import processing_queries as pq
import generate_inverted_list as gil
import indexer
import searcher
import logging
import sys


# Configuramos o arquivo que será o log do sistema
logging.basicConfig(filename='../results/search_mechanism.log', filemode='w',format='%(asctime)s - %(message)s', level=logging.INFO, force=True)
logging.info("Log created.")

# Início da execução do módulo 'processando consultas'
logging.info("Module processing_queries started.")
read, queries, expected = pq.read_config("pc.cfg")
# Lendo arquivo com consultas
xml_root = pq.get_xml_root(read)
# Escrevendo arquivos com consultas e resultados esperados, respectivamente
pq.get_queries_file(queries, xml_root)
pq.get_expected_file(expected, xml_root)
# Fim da execução do módulo
logging.info("Module processing_queries finished execution.")


# Início da execução do módulo 'gerador lista invertida'
logging.info("Module generate_inverse_list started.")
read_files, write_file = gil.read_config_file("gli.cfg")
# Gerando arquivo da lista invertida
gil.get_tokens_file(read_files, write_file)
# Fim da execução do módulo
logging.info("Module processing_queries finished execution.")


# Início da execução do módulo 'indexador'
logging.info("Module indexer started.")
# Escolha do usuário entre usar o tf normalizado ou não
normalized = input("tf normalized [ y / n ]? ")
if normalized.lower() == "y":
    type_tf = "tfn"
else:
    type_tf = "tf"
tokens, model = indexer.read_config_file("index.cfg")
# Gerando modelo através da matriz termo documento que foi construída com a lista invertida
indexer.save_model(model, tokens, type_tf)
# Fim da execução do módulo
logging.info("Module indexer finished execution.")


# Início da execução do módulo 'buscador'
logging.info("Module searcher started.")
model_file, queries_file, results_file = searcher.read_config_file("busca.cfg")
# Lê o modelo na memória
model = searcher.get_model(model_file)
# Lê as consultas na memória
queries = searcher.get_queries(queries_file)
# Usa as consultas e o modelo para gerar o ranking de documentos que mais se aproximam das consultas
ranking = searcher.get_ranking(model, queries)
# Gera o arquivo de resultados com o arquivo de ranking gerado
searcher.get_results(results_file, ranking)
# Fim da execução do módulo
logging.info("Module searcher finished exectuion.")
