import processing_queries as pq
import generate_inverted_list as gil
import indexer
import searcher
import logging
import sys


# Início da execução do módulo 'processando consultas'
pq.start_exec()
read, queries, expected = pq.read_config("pc.cfg")
# Lendo arquivo com consultas
xml_root = pq.get_xml_root(read)
# Escrevendo arquivos com consultas e resultados esperados, respectivamente
pq.get_queries_file(queries, xml_root)
pq.get_expected_file(expected, xml_root)
# Fim da execução do módulo
pq.finish_exec()


# Início da execução do módulo 'gerador lista invertida'
gil.start_exec()
read_files, write_file, stemmer = gil.read_config_file("gli.cfg")
# Gerando arquivo da lista invertida
gil.get_tokens_file(read_files, write_file, stemmer)
# Fim da execução do módulo
gil.finish_exec()


# Início da execução do módulo 'indexador'
indexer.start_exec()
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
indexer.finish_exec()


# Início da execução do módulo 'buscador'
searcher.start_exec()
model_file, queries_file, results_file, stemmer = searcher.read_config_file("busca.cfg")
# Lê o modelo na memória
model = searcher.get_model(model_file)
# Lê as consultas na memória
queries = searcher.get_queries(queries_file, stemmer)
# Usa as consultas e o modelo para gerar o ranking de documentos que mais se aproximam das consultas
ranking = searcher.get_ranking(model, queries)
# Gera o arquivo de resultados com o arquivo de ranking gerado
searcher.get_results(results_file, ranking)
# Fim da execução do módulo
searcher.finish_exec()
