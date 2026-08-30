# Projeto Final: Inteligência Artificial
**Disciplina:** Inteligência Artificial  
**Professor:** Prof. Rafael Bressan  
**Aluna:** Ana Larissa Dias  

Repositório dedicado ao desenvolvimento e entrega da atividade final da disciplina de Inteligência Artificial. O projeto está estruturado em três módulos independentes, cobrindo algoritmos de busca, meta-heurísticas para o TSP e aprendizado de máquina supervisionado.

## Estrutura do Repositório

```text
.
├── 1_sistemas_de_busca/
│   ├── a_star.py
│   ├── main.py
│   ├── maze.py
│   ├── read_matrix.py
│   ├── robot.py
│   ├── visualization.py
│   ├── Readme.md
│   ├── requirements.txt
│   ├── data/
│   └── results/
├── 2_meta_heurística_no_tsp/
│   ├── baseline.py
│   ├── projeto_final_TSP_ACO.py
│   ├── projeto_final_TSP_GA.py
│   ├── run_experiments.py
│   ├── read_tsp.py
│   ├── Readme.md
│   ├── requirements.txt
│   ├── data/
│   └── results/
├── 3_aprendizado_supervisionado/
│   ├── load_data.py
│   ├── metrics.py
│   ├── plots.py
│   ├── run_baseline.py
│   ├── run_refinements.py
│   ├── split_data.py
│   ├── README.md
│   ├── articles/
│   ├── data/
│   └── results/
└── README.md
```

## Módulos do Projeto
### 1. Sistemas de Busca e Heurísticas (1_sistemas_de_busca)
Implementação e comparação de estratégias de pathfinding em um labirinto matricial (256x256). Avalia o desempenho de um agente guloso com memória em pilha contra o algoritmo A* utilizando a heurística de Chebyshev para garantir a admissibilidade.

### 2. Meta-heurísticas no Caixeiro Viajante (2_meta_heurística_no_tsp)
Otimização combinatória para o Traveling Salesperson Problem (TSP) testada em instâncias reais da TSPLIB. Compara o desempenho de um Algoritmo Genético (cruzamento OX e mutação por inversão) contra a Otimização por Colônia de Formigas (ACO), ambos limitados por um orçamento de tempo estrito, utilizando o Vizinho Mais Próximo como baseline.

### 3. Aprendizado Supervisionado (3_aprendizado_supervisionado)
Pipeline completo de classificação para credit scoring utilizando um dataset com quase 220 mil registros desbalanceados e dados anonimizados. A escolha do XGBoost como modelo vencedor foi fundamentada por levantamento bibliográfico anexo (pasta articles/), aplicando otimização do limiar de decisão via Índice de Youden para maximizar a detecção da classe minoritária.

## Execução e Dependências
Cada um dos três módulos funciona de forma independente e possui seu próprio arquivo README.md detalhando os pré-requisitos, instruções de instalação (via requirements.txt quando aplicável) e comandos exatos para execução dos pipelines e geração de resultados/gráficos.

Para reproduzir os experimentos, navegue até a pasta do módulo desejado e siga as instruções locais.