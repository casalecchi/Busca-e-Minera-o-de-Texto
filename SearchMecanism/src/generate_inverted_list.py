import os
import logging
import numpy as np
from datetime import datetime
from xml.etree import ElementTree as ET
import nltk
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


def read_config_file(config_file):
    """"Retorna os paths dos arquivos de leitura e de escrita. Lê o arquivo de configuração do módulo. Os path dos
    arquivos de leitura são passados em uma lista na primeira posição do retorno."""

    logging.basicConfig(filename='../results/gli.log', filemode='a',format='%(asctime)s - %(message)s', level=logging.INFO, force=True)
    logging.info("Started reading the configuration file.")
    read_files = []
    write_file = "../results/"
    cfg_path = "../data/" + config_file

    with open(cfg_path, "r") as config_file:
        for line in config_file.readlines():
            instruction, filename = line.split("=")
            filename = filename.strip()
            
            if instruction == "LEIA":
                file_path = os.path.join("../data/CysticFibrosis", filename)
                read_files.append(file_path)
            elif instruction == "ESCREVA":
                write_file += filename

    logging.info("Finished reading the configuration file.")
    
    return (read_files, write_file)


def get_recordnum_text(file):
    """É retornado um dicionário contendo o número RECORDNUM como chave e o texto presente no RECORD como seu valor
    correspondente. É necessário passar o path de um arquivo XML."""

    logging.basicConfig(filename='../results/gli.log', filemode='a',format='%(asctime)s - %(message)s', level=logging.INFO, force=True)
    logging.info(f"Started creating the dict record_num: text from file: {file}")
    start_time = datetime.now()
    
    xml_file = ET.parse(file)
    xml_root = xml_file.getroot()
    
    recordnum_text = {}
    total_records = 0
    
    for record in xml_root:
        total_records += 1
        text = ""
        for element in record:
            if element.tag == "RECORDNUM":
                record_num = int(element.text)
            elif element.tag == "ABSTRACT" or element.tag == "EXTRACT":
                text = element.text.upper()
        recordnum_text[record_num] = text
    
    time_taken = datetime.now() - start_time
    logging.info(f"Finished creating the dict record_num: text from file: {file}. {total_records} records in file. Time taken: {time_taken}s")
    
    # Retorna dicionário na forma RecordNum: Texto para cada arquivo passado
    return recordnum_text


def preproccess_text(text):
    """Retorna o texto inserido como parâmetro pré-processado. Em suma, o texto prmeiro é tokenizado. Depois é removido suas stopwords, palavras 
    que contenham caracteres que não sejam letras e palavras menores que 3 caracteres. Por fim é aplicado um stemming em todas as palavras restantes."""
    
    # Texto é tokenizado
    tokens = wordpunct_tokenize(text)
    stop_en = stopwords.words("english")

    # Remove os símbolos
    filtered_text = []
    for word in tokens:
        # Remove stopwords
        if word.lower() in stop_en:
            continue
        # Remove palavras que contenham caracteres além de letras
        elif not word.isalpha():
            continue
        # Remova palavras com tamanho menor que 3 caracteres
        elif len(word) < 3:
            continue
        # Palavra vai "entrar" mas antes vamos aplicar um stemming
        else:
            stemmer = PorterStemmer()
            word_stemmed = stemmer.stem(word)
            filtered_text.append(word_stemmed.upper())
    
    return filtered_text


def word_frequency(text, record_num):
    """Retorna a lista invertida para um único texto passado."""

    # Primeiro pré-processamos o texto
    tokenized_text = preproccess_text(text)
    frequency_dict = {}
    
    for word in tokenized_text:
        keys = list(frequency_dict.keys())
        if word in keys:
            frequency_dict[word].append(record_num)
        else:
            frequency_dict[word] = [record_num]
            
    # Dicionário retornado terá seguinte forma - {palavra: [record_num, record_num, ...]}
    return frequency_dict


def get_inverted_list(read_files):
    """Geração de uma única lista invertida juntando todos os arquivos de leitura. É passado uma lista contendo os paths dos arquivos XML
    de leitura. A lista é retornada como um dicionário, contendo uma palavra como chave e uma lista com os documentos que aparecessem como valor."""

    logging.basicConfig(filename='../results/gli.log', filemode='a',format='%(asctime)s - %(message)s', level=logging.INFO, force=True)
    logging.info(f"Started creating the inverted list.")
    times = np.array([])
    total_files = 0
    inverted_list = {}
    
    for file in read_files:
        start_time = datetime.now()
        total_files += 1 

        # Faz primeiro o dicionário do arquivo, com os record_num: textos
        file_records = get_recordnum_text(file)
        file_record_nums = list(file_records.keys())
        
        # Depois, pega cada record
        for record_num in file_record_nums:
            # Faz o dicionário de frequência de um record
            record_dict = word_frequency(file_records[record_num], record_num)
            
            # Pegar o dicionário de frequência e juntar no geral
            used_tokens = list(record_dict.keys())
            # Atualiza ou cria os tokens presentes no record no dicionário geral
            for token in used_tokens:
                # Pega o que já tinha, ou [] caso não exista
                previous_records = inverted_list.get(token, [])
                if previous_records == []:
                    # Cria
                    inverted_list[token] = record_dict[token]
                else:
                    #Atualiza
                    inverted_list[token] += record_dict[token]

        time_taken = datetime.now() - start_time
        times = np.append(times, [time_taken])
    
    mean = np.mean(times)
    logging.info(f"{total_files} files procesed. Average time: {mean}s.")
    logging.info(f"Finished creating the inverted list.")
    
    return inverted_list

def get_tokens_file(read_files, path):
    """É criado o arquivo de escrita, contendo a lista invertida. Passamos a lista com os arquivos de leitura e o path do arquivo que 
    será escrito."""

    logging.basicConfig(filename='../results/gli.log', filemode='a',format='%(asctime)s - %(message)s', level=logging.INFO, force=True)
    logging.info(f"Started creating the inverted list file.")
    
    with open(path, 'w') as w_file:
        inverted_list = get_inverted_list(read_files)
        tokens = list(inverted_list.keys())
        w_file.write("Token;Appearance\n")
        
        for token in tokens:
            w_file.write(f"{token};{inverted_list[token]}\n")
    
    logging.info(f"Finished creating the inverted list file.")


def start_exec():
    logging.basicConfig(filename='../results/gli.log', filemode='a',format='%(asctime)s - %(message)s', level=logging.INFO, force=True)
    logging.info("Module generate_inverted_list started.")


def finish_exec():
    logging.basicConfig(filename='../results/gli.log', filemode='a',format='%(asctime)s - %(message)s', level=logging.INFO, force=True)
    logging.info("Module generate_inverted_list finished execution.")


# Configuramos o arquivo que será o log do módulo e baixamos dependências do nltk
logging.basicConfig(filename='../results/gli.log', filemode='w',format='%(asctime)s - %(message)s', level=logging.INFO, force=True)
logging.info("Log created.")
logging.info("Downloading 'punkt' and 'stopwords' from nltk-data...")
nltk.download('punkt')
nltk.download('stopwords')
logging.info("Finished downloading 'punkt' and 'stopwords'.")
