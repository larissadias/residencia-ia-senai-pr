# Inteligência Artificial - Projetos e Implementações

Este repositório centraliza os projetos, algoritmos e desafios desenvolvidos durante a disciplina de Inteligência Artificial. O objetivo principal é documentar a transição de conceitos acadêmicos e teóricos para implementações práticas, aplicáveis na resolução de problemas complexos de otimização, busca e aprendizado de máquina.

Todos os códigos foram desenvolvidos com foco em estruturação clara, eficiência computacional e ausência de dependência excessiva de bibliotecas de "caixa preta", priorizando a construção nativa de grafos, heurísticas e modelos.

---

## 🗂️ Índice de Projetos
### Módulo I: Inteligência Artificial

As pastas abaixo refletem a progressão da disciplina, organizados por área de estudo da Inteligência Artificial. Cada diretório contém seu próprio `README.md` com instruções detalhadas de execução e os resultados obtidos.

### 1. Buscas e Agentes Inteligentes
Projetos focados em algoritmos de busca (cega e heurística) para resolução de problemas de espaço de estados.
* **Labirinto Matricial (Busca Heurística)**: Agente inteligente navegando em labirintos NxN com métricas Euclidiana, Manhattan e Chebyshev.
* **Puzzle de 8 peças**: Resolução de quebra-cabeças 3x3.
* **Problema das N-Rainhas**: Posicionamento de rainhas em tabuleiro NxN livre de ataques.

### 2. Teoria dos Grafos e Caminhos Mínimos
Representação e manipulação de grafos estruturados nativamente.
* **Algoritmo de Dijkstra**: Estruturação via POO (Programação Orientada a Objetos) para encontrar o menor caminho e gerar matrizes de distâncias.

### 3. Decisões Adversariais (Teoria dos Jogos)
Implementação de agentes capazes de tomar decisões em ambientes competitivos de soma zero.
* **Jogo da Velha (Algoritmo MinMax)**: IA imbatível calculando a próxima jogada ideal através de árvore de jogos.

### 4. Metaheurísticas e Otimização
Abordagens avançadas para resolver o Problema do Caixeiro Viajante (TSP) e outros problemas NP-Difíceis.
* **Algoritmos Genéticos (GA)**
* **Simulated Annealing**
* **Otimização por Enxame de Partículas (PSO)**
* **Otimização por Colônia de Formigas (ACO)**

### 5. Lógica Fuzzy
Sistemas de inferência baseados em regras para lidar com imprecisão e graus de verdade.
* **Sistema de Mesada**: Avaliação inteligente usando lógica nebulosa.
* **Qualidade do Leite**: Sistema Fuzzy para classificação e avaliação de qualidade.

### 6. Machine Learning (Aprendizado Supervisionado)
Modelos preditivos treinados sobre bases de dados estruturadas (ex: base Diabetes).
* **K-Nearest Neighbors (KNN)**: Implementação e estratégias de otimização de hiperparâmetros.
* **Árvores de Decisão**: Construção, treinamento e aperfeiçoamento de desempenho dos modelos.

---

## 🛠️ Tecnologias e Ferramentas Utilizadas

* **Linguagem:** Python 3.x
* **Manipulação e Matemática:** NumPy, Math
* **Visualização:** Matplotlib
* **Paradigmas:** Programação Orientada a Objetos (POO), Estruturas de Dados Nativas (Pilhas, Filas, Dicionários, Matrizes).

---

## 🚀 Como Explorar

Cada subdiretório neste repositório é um projeto independente. Para testar qualquer um dos algoritmos:
1. Clone este repositório: `git clone https://github.com/seu-usuario/nome-do-repositorio.git`
2. Navegue até a pasta do módulo desejado.
3. Leia o arquivo `README.md` local da pasta para instalar as dependências e executar o código fonte.
