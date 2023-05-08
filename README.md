# Busca e Mineração deTexto - 2023.1

## Search Mechanism

O código foi feito contendo três pastas: `src`, `data` e `results`. 

* A primeira contém o código fonte para os quatro módulos desenvolvidos. Além disso, foi feito um notebook no Jupyter Notebook para melhor visualização do que os módulos estão fazendo quando são chamados. Neste notebook também é comentado sobre a forma de implementação dos módulos. Para mais informações sobre a implementação, é possível checar no próprio código fonte, onde temos os códigos comentados. 

* Na pasta `data` é onde temos os dados que serão utilizados na execução do mecanismo de busca. Portanto, contém o dataset disponibilizado, CysticFibrosis, e também os arquivos de configuração dos módulos conforme o enunciado do trabalho. 

* Já na última pasta, `results`, é onde todo arquivo que for gerado pela execução será armazenado. Portanto, teremos o log do mecanismo de busca, além de alguns arquivos no formato CSV.

Para o desenvlvimento do código, foi utilizado o miniforge3 com os pacotes listados no arquivo `requirements.txt` presente no diretório `SearchMechanism`.

### Guia de uso

Primeiro é necessário instalar todas as dependências. Elas estão presentes no arquivo `requirements.txt` no diretório `SearchMechanism`. Para realizar isso utilizando uma venv:

```shell
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

Após a instalação, basta rodar o arquivo `main.py`, estando no diretório `src`, da seguinte maneira:

```shell
python3 main.py
```

Durante a execução, será necessário indicar no próprio terminal se o parâmetro `tf` será utilizado de maneira normalizada ou não. Para isso basta seguir as instruções que serão printadas no terminal.

A geração do modelo, nos testes realizados, levou em torno de 6 minutos para ser completada. Uma barra de progresso com a estimativa de tempo foi utilizada para que seja mais fácil essa visualização. 