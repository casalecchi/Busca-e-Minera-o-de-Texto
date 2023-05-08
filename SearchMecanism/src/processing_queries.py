import logging
import numpy as np
from datetime import datetime
from xml.etree import ElementTree as ET


def read_config(config_file):
    logging.info("Started reading the configuration file.")

    read_file = "../data/CysticFibrosis/"
    queries_file = "../results/"
    expected_file = "../results/"
    cfg_path = "../data/" + config_file

    with open(cfg_path, "r") as config_file:
        for line in config_file.readlines():
            instruction, filename = line.split("=")
            filename = filename.strip()

            if instruction == "LEIA":
                read_file += filename
            elif instruction == "CONSULTAS":
                queries_file += filename
            elif instruction == "ESPERADOS":
                expected_file += filename
                
    logging.info("Finished reading the configuration file.")
    return (read_file, queries_file, expected_file)


def get_xml_root(path):
    logging.info("Parsing the xml queries file.")
    xml_file = ET.parse(path)
    xml_root = xml_file.getroot()
    logging.info("XML file parsed.")
    return xml_root


def get_queries_file(path, xml_root):
    logging.info("Starting the generation of queries csv file.")
    with open(path, 'w') as queries:
        queries.write("QueryNumber;QueryText\n")
        queries_read = 0
        times = np.array([])
        for query in xml_root:
            start_time = datetime.now()
            queries_read += 1
            query_number = ""
            query_text = ""
            for element in query:
                if element.tag == "QueryNumber":
                    query_number = int(element.text)
                elif element.tag == "QueryText":
                    query_text = element.text.upper()
                    query_text = query_text.replace('\n  ', '')
                    query_text = query_text.replace(';', '')
            queries.write(f"{query_number};{query_text}")
            time_taken = datetime.now() - start_time
            times = np.append(times, [time_taken])

    mean = np.mean(times)
    logging.info(f"{queries_read} queries processed.")
    logging.info(f"Mean of the time each query has taken to be processedin queries file: {mean}s")
    logging.info("Queries csv file generated.")


def get_expected_file(path, xml_root):
    logging.info("Starting the generation of expected csv file.")
    with open(path, 'w') as expected:
        expected.write("QueryNumber;DocNumber;DocVotes\n")
        times = np.array([])
        for query in xml_root:
            start_time = datetime.now()
            query_number = ""
            for element in query:
                if element.tag == "QueryNumber":
                    query_number = int(element.text)
                elif element.tag == "Records":
                    for item in element:
                        doc_number = int(item.text)
                        score = item.attrib['score'].replace('0', '')
                        doc_votes = len(score)
                        expected.write(f"{query_number};{doc_number};{doc_votes}\n")
            time_taken = datetime.now() - start_time
            times = np.append(times, [time_taken])
    
    mean = np.mean(times)
    logging.info(f"Mean of the time each query has taken to be processed in expected file: {mean}s")
    logging.info("Expected csv file generated.")
