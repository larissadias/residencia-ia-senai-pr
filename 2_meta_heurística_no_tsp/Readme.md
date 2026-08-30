# Projeto Final Parte 2: Meta-heurística no Problema do Caixeiro Viajante (TSP)
**Disciplina:** Inteligência Artificial
**Professor:** Prof. Rafael Bressan
**Aluna:** Ana Larissa Dias

Comparação entre **Algoritmo Genético** e **Otimização por Colônia de Formigas**
em três instâncias reais da TSPLIB, usando a heurística do vizinho mais próximo
como baseline e a rota ótima publicada como piso de referência.


## Estrutura dos arquivos

| Arquivo | Papel |
|---|---|
| `run_experiments.py` | Executa a comparação completa e gera as evidências |
| `projeto_final_TSP_GA.py` | Algoritmo Genético |
| `projeto_final_TSP_ACO.py` | Colônia de Formigas |
| `baseline.py` | Heurística do vizinho mais próximo |
| `read_tsp.py` | Leitura das instâncias TSPLIB e matriz de distâncias |
| `data/` | Instâncias e rotas ótimas |

As instâncias vêm da TSPLIB, disponível em
`https://www.math.uwaterloo.ca/tsp/data/usa/index.html`.

## Pré-requisitos

```bash
pip install matplotlib
```

## Como rodar

```bash
# Comparação completa nas três instâncias
python run_experiments.py

```

Saídas: `resultados.csv`, `resultados_por_semente.csv` e `convergencia.png`.


## Representação da solução

Cada solução é uma **permutação dos índices das cidades**. A rota é um ciclo
fechado, então `[0, 3, 1, 2]` representa o trajeto 0 → 3 → 1 → 2 → 0.

No Algoritmo Genético essa permutação é o cromossomo; na Colônia de Formigas
é o trajeto percorrido por uma formiga.

O codigo garante que toda solução seja válida por construção, onde os
operadores apenas reordenam os índices, de modo que nenhuma cidade é repetida ou
omitida. Por isso não é necessário aplicar penalidades na função objetivo.

Além disso, os indivíduos são listas de inteiros, não de nomes de
cidades. Na atividade da Aula 05, com 6 cidades, a rota era `['A', 'C', 'B']` e o
custo exigia `CITIES.index()` a cada consulta, uma busca linear. Mas, som 442
cidades e milhares de avaliações por geração, isso tornaria a função objetivo
centenas de vezes mais lenta, porém com índices o acesso à matriz é direto.


## Função objetivo

A distância total da rota, incluindo o retorno à origem:

```python
def calculates_distance(route, distance_matrix):
    distance = 0
    for i in range(len(route)):
        current_city = route[i]
        next_city = route[(i + 1) % len(route)]
        distance += distance_matrix[current_city][next_city]
    return distance
```

O operador de módulo faz a última cidade se ligar à primeira, fechando o ciclo.
Quanto menor, melhor.

### A transformação para o fitness

A seleção por torneio do Algoritmo Genético escolhe o indivíduo de maior
aptidão, mas a função objetivo deve ser minimizada. O fitness é definido
como o inverso da distância:

```python
def fitness(route, distance_matrix):
    return 1.0 / calculates_distance(route, distance_matrix)
```

A função objetivo é a distância, e o fitness é a transformação que converte 
minimização em maximização.

### A Matriz de Distâncias

As instâncias TSPLIB não trazem a matriz pronta, e sim as coordenadas das
cidades. A matriz é calculada com a métrica `EUC_2D`, distância
euclidiana arredondada para o inteiro mais próximo.

## Estratégia de Busca

### Algoritmo Genético

| Componente | Escolha |
|---|---|
| População | 100 indivíduos, permutações aleatórias |
| Seleção | Torneio de 3 |
| Cruzamento | OX |
| Mutação | Inversão, com taxa de 0,2 |
| Elitismo | O melhor indivíduo passa intacto |


O efeito dos operadores foi medido no berlin52 (ótimo 7.542, 2.000 gerações):

| Cruzamento | Mutação | Custo | Gap |
|---|---|---|---|
| PMX | troca | 9.725 | 28,9% |
| PMX | inversão | 8.168 | 8,3% |
| OX | troca | 8.255 | 9,5% |
| **OX** | **inversão** | **8.117** | **7,6%** |

Os operadores da atividade original (PMX + troca) davam 28,9% de gap. A mutação
sozinha explica a maior parte do ganho.

**Exploração × intensificação:** A taxa de mutação e o tamanho do torneio
controlam a exploração, mantendo diversidade; o elitismo garante intensificação,
preservando a melhor solução; o cruzamento atua nos dois sentidos.

Os dois operadores continuam disponíveis no código, selecionáveis pelas
constantes `CROSSOVER_TYPE` e `MUTATION_TYPE`, para que a comparação seja
reproduzível.

### Colônia de Formigas

| Componente | Escolha |
|---|---|
| Formigas | 50 por iteração |
| ALPHA (peso do feromônio) | 1,0 |
| BETA (peso da distância) | 5,0 |
| Taxa de evaporação | 0,1 |
| Feromônio inicial | 1,0 |

Cada formiga parte de uma cidade aleatória e constrói a rota escolhendo a
próxima cidade de forma probabilística, com peso proporcional a
`feromônio^ALPHA × (1/distância)^BETA`. Ao final da iteração o feromônio evapora
e as formigas depositam nas arestas que usaram, na proporção inversa ao custo da
rota.

**Por que BETA = 5?** Este foi o ajuste de maior efeito no bloco. Com `BETA = 3`,
o gap na instância pesada era de 80,7%; com `BETA = 5`, caiu para 31,7%.


## Instâncias

Três níveis de complexidade, todas `EUC_2D` e com rota ótima publicada.

| Instância | Cidades | Ótimo conhecido |
|---|---|---|
| `pr76` | 76 | 108.159 |
| `tsp225` | 225 | 3.861 |
| `pcb442` | 442 | 50.778 |

---

## Comparação por orçamento de tempo

As duas técnicas recebem o mesmo tempo, não o mesmo número de iterações.

Contar iterações seria injusto. Uma iteração do ACO faz 50 formigas
construírem rotas com escolha probabilística sobre todas as cidades restantes com
custo O(n²) por formiga mais a atualização da matriz de feromônio. Uma geração
do Algoritmo Genético aplica cruzamentos e mutações, que são O(n).

O mesmo número de iterações significaria esforços computacionais muito
diferentes. PJá com orçamento de tempo, cada técnica mostra o que consegue alcançar
com o mesmo recurso.

| Instância | GA | ACO |
|---|---|---|
| Leve (76) | 10.280 gerações | 1.065 iterações |
| Intermediária (225) | 3.818 gerações | 118 iterações |
| Pesada (442) | 1.359 gerações | **30 iterações** |



## Resultados

Orçamento de 30 segundos por execução, 3 sementes por técnica.

| Instância | Ótimo | Vizinho mais próximo | AG (média) | ACO (média) |
|---|---|---|---|---|
| Leve (76) | 108.159 | 153.462 (**41,9%**) | **112.849 (4,3%)** | 115.807 (7,1%) |
| Intermediária (225) | 3.861 | 4.722 (**22,3%**) | 6.146 (59,2%) | **4.313 (11,7%)** |
| Pesada (442) | 50.778 | **61.979 (22,1%)** | 265.146 (422,2%) | 66.868 (31,7%) |

O vizinho mais próximo roda em 0,2 ms, 1,7 ms e 8,4 ms respectivamente.

### Consistência entre execuções

| Instância | Técnica | Melhor | Média | Pior | Desvio |
|---|---|---|---|---|---|
| Leve | AG | 110.773 | 112.849 | 114.288 | 1.842 |
| Leve | ACO | 114.506 | 115.807 | 116.850 | 1.193 |
| Intermediária | AG | 5.022 | 6.146 | 7.310 | **1.145** |
| Intermediária | ACO | 4.289 | 4.313 | 4.331 | **22** |
| Pesada | AG | 260.212 | 265.146 | 272.687 | 6.634 |
| Pesada | ACO | 66.250 | 66.868 | 68.021 | 999 |

-
## Conclusões

### Nenhuma técnica vence sempre

Com 76 cidades o Algoritmo Genético vence; com 225 e 442, a Colônia de Formigas.
A resposta de qual é melhor depende do tamanho do problema.


### Na instância pesada, as duas perdem para o baseline

 O vizinho mais próximo resolve o `pcb442` com 22,1% de gap em 8 milissegundos, 
 enquanto as duas metaheurísticas gastam 30 segundos e ficam piores.

Isso mostra que o orçamento de tempo é insuficiente para o
tamanho do problema. Com 442 cidades, 30 iterações do ACO e 1.359 gerações do
AG são muito pouco. Uma heurística construtiva gulosa, que resolve em um único
passo, leva vantagem nesse cenário.

### O ACO é mais consistente

Na instância intermediária o AG variou entre 5.022 e 7.310 conforme a semente,
desvio de 1.145. O ACO variou entre 4.289 e 4.331, desvio de 22, cinquenta
vezes menor.

### O melhor resultado aparece cedo no ACO e tarde no AG

Na instância pesada, o ACO encontra seu melhor por volta da iteração 26 de 30,
mas na leve, encontra na iteração 500 de 1.065 e depois estagna. O AG, ao
contrário, costuma melhorar até o fim do orçamento.Isso indica que o ACO converge 
rápido e depois satura, enquanto o AG ainda
tinha margem quando o tempo acabou.
