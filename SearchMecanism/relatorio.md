# Relatório do Trabalho de Avaliação do MEcanismo de Busca

## 0. Construção de gráficos e obtenção dos dados

Todo o processo de construção dos gráficos, assim como a estruturação dos dados que os compõem, estão especificados em um notebook do Jupyter presente no diretório `src`, de nome `Avaliacao.ipynb`. Dentro dele temos as criações para os gráficos dos resultados do mecanismo de busca utilizando stemmer. Os gráficos para o mecanismo que não utilizam o stemmer, foram criados utilizando o mesmo notebook, mas apenas mudando o arquivo que é passado em seu início. 

Como era esperado, os arquivos CSV que foram utilizados para fazer os gráficos estão no diretório `avalia`, assim como eles próprios. O README do mecanismo foi atualizado indicando a mudança que foi implementada para utilizar ou não o stemmer. Mais detalhes sobre a criação das métricas de avaliação podem ser encontradas dentro do próprio notebook.

## 1. Gráfico de 11 pontos de precisão e recall

* Stemmer

![11pontos-stemmer](avalia/11pontos-stemmer-1.png)

* NoStemmer

![11pontos-nostemmer](avalia/11pontos-nostemmer-2.png)

## 2. $F_1$ score

* Stemmer

![f1-stemmer](avalia/f1-stemmer-3.png)

* NoStemmer

![f1-nostemer](avalia/f1-nostemmer-4.png)

## 3. Precision@5

* Stemmer

![precision@5-stemmer](avalia/precision@5-stemmer-5.png)

* NoStemmer

![precision@5-nostemmer](avalia/precision@5-nostemmer-6.png)

## 4. Precision@10

* Stemmer

![precision@10-stemmer](avalia/precision@10-stemmer-7.png)

* NoStemmer

![precision@10-nostemmer](avalia/precision@10-nostemmer-8.png)

## 5. Histograma Comparativo de R-Precision

![r-precision](avalia/r-precision-comparativo-9.png)

## 6. MAP

* Stemmer

![map-stemmer](avalia/map-stemmer-10.png)

* NoStemmer

![map-nostemmer](avalia/map-nostemmer-11.png)

## 7. MRR

* Stemmer

![mrr-stemmer](avalia/mrr-stemmer-12.png)

* NoStemmer

![mrr-nostemmer](avalia/mrr-nostemmer-13.png)

## 8. Discounted Cumulative Gain

* Stemmer

![dcg-stemmer](avalia/dcg-stemmer-14.png)

* NoStemmer

![dcg-nostemmer](avalia/dcg-nostemmer-15.png)

## 9. Normalized Discounted Cumulative Gain

* Stemmer

![ndcg-stemmer](avalia/ndcg-stemmer-16.png)

* NoStemmer

![ndcg-nostemmer](avalia/ndcg-nostemmer-17.png)



