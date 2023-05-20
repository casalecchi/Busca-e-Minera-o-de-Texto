# Search Mechanism

## Estruturação do código - Trabalho 1

O código foi feito contendo três pastas: `src`, `data` e `results`. 

* A primeira contém o código fonte para os quatro módulos desenvolvidos. Além disso, foi feito um notebook no Jupyter Notebook para melhor visualização do que os módulos estão fazendo quando são chamados. Neste notebook também é comentado sobre a forma de implementação dos módulos. Para mais informações sobre a implementação, é possível checar no próprio código fonte, onde temos os códigos comentados. 

* Na pasta `data` é onde temos os dados que serão utilizados na execução do mecanismo de busca. Portanto, contém o dataset disponibilizado, CysticFibrosis, e também os arquivos de configuração dos módulos conforme o enunciado do trabalho. 

* Já na última pasta, `results`, é onde todo arquivo que for gerado pela execução será armazenado. Portanto, teremos o log do mecanismo de busca, além de alguns arquivos no formato CSV.

Para o desenvlvimento do código, foi utilizado o miniconda com os pacotes listados no arquivo `requirements.txt`.

Também é disponibilzado o arquivo `modelo.txt` contendo a descrição do formato do modelo vetorial que é criado pelo programa.

### Guia de uso

Primeiro é necessário instalar todas as dependências presentes em `requirements.txt`. É recomendo realizar isso utilizando o próprio comando conda:

```shell
conda create --name <environment_name> --file requirements.txt
```

Após a instalação, basta rodar o arquivo `main.py`, estando no diretório `src`, da seguinte maneira:

```shell
python3 main.py
```

Durante a execução, será necessário indicar no próprio terminal se o parâmetro `tf` será utilizado de maneira normalizada ou não. Para isso basta seguir as instruções que serão printadas no terminal.

A geração do modelo, nos testes realizados, levou em torno de 6 minutos para ser completada. Uma barra de progresso com a estimativa de tempo foi utilizada para que seja mais fácil essa visualização. 

## Avaliação do modelo - Trabalho 2

Nesse segundo trabalho é pedido para que o modelo possa ser avaliado utilizando diferentes métricas. Antes de irmos até elas, temos algumas mudanças importantes no funcionamento do mecanismo. Ambas ocorrem detnro do arquivo de configuração de dois módulos. 

A primeira mudança é presente no arquivo de configuração do módulo "Gerar Lista Invertida" (`gli.cfg`). Na primeira linha, onde teremos a indicação se será utilizado um stemmer ou não. Portanto as opções para essa linha são utilizar a palavra "STEMMER" ou "NOSTEMMER".

A segunda mudança se dá no arquivo de configuração do módulo "Buscador" (`busca.cfg`). Aqui indicaremos se o texto das consultas passarão ou não por um stemmer. Isso será feito adicionando no nome de arquivo de resultados um "-stemmer", em caso positivo, ou "-nostemmer", em caso negativo.