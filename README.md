# Projeto Final: Rede Neural Artificial (INE-5664)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)

## Grupo 14

| Integrante | Matrícula |
| :--- | :--- |
| Francisco Camerini | 22100898 |
| Lucas Almeida Lazarini | 22104164 |
| Luiz Felipe E. B. Scheidt | 22100914 |

## Descrição do Projeto

Este projeto consiste na implementação de uma Rede Neural Artificial (RNA) utilizando Python. O objetivo é demonstrar o entendimento dos conceitos centrais de uma rede neural, aplicando a implementação a um dataset de defeitos de fabricação para resolver três tarefas distintas de machine learning.

### Principais Implementações

- Estrutura completa da rede (camadas, pesos e biases).
- Três funções de ativação: Sigmoid, ReLU e Tanh.
- Duas funções de perda: Mean Squared Error (MSE) e Binary Cross-Entropy (BCE).
- Algoritmo de retropropagação (*backpropagation*) e otimização por Gradiente Descendente.

## Análise Realizada

A rede neural foi avaliada em três cenários, conforme os requisitos do projeto:

1.  **Regressão:** Previsão do valor contínuo da taxa de defeitos (`DefectRate`).
2.  **Classificação Binária:** Predição da ocorrência de um defeito (`DefectStatus`).
3.  **Classificação Multiclasse:** Categorização da taxa de defeitos em três níveis distintos (`DefectLevel`).

## Estrutura do Projeto

A estrutura do projeto foi simplificada para facilitar a execução e a análise em um único ambiente de notebook:

```
.
├── data/
│   └── manufacturing_defects.csv
├── Projeto_Rede_Neural.ipynb
└── requirements.txt
```

- **`data/`**: Contém o conjunto de dados utilizado.
- **`Projeto_Rede_Neural.ipynb`**: O Jupyter Notebook que contém toda a implementação, análise e documentação do projeto.
- **`requirements.txt`**: Arquivo com as dependências de bibliotecas Python necessárias para executar o projeto.

## Como Executar

Para executar o notebook e replicar os resultados, siga os passos abaixo.

### 1. Clonar o Repositório

```bash
git clone https://github.com/FranciscoCamerini/INE-5664.git
cd INE-5664
```

### 2. Criar e Ativar um Ambiente Virtual

**No macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**No Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Instalar as Dependências

Com o ambiente virtual ativado, instale as bibliotecas necessárias:

```bash
pip install -r requirements.txt
```

### 4. Iniciar o Jupyter e Executar o Notebook

Finalmente, inicie o ambiente Jupyter para abrir e executar o notebook.

```bash
jupyter lab
```

Ou, se preferir a interface clássica:

```bash
jupyter notebook
```