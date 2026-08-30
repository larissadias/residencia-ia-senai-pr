# Projeto Final Parte 1: Sistemas de Busca com o Labirinto Matricial do Bressan
**Disciplina:** Inteligência Artificial
**Professor:** Prof. Rafael Bressan
**Aluna:** Ana Larissa Dias

Comparação entre o agente com heurística gulosa da Atividade 1 da disciplina e 
algoritmo de busca informada A\*, usando busca em largura como piso teórico
de referência.

## Estrutura dos arquivos

| Arquivo | Papel |
|---|---|
| `main.py` | Executa a comparação, monta a tabela e gera as figuras |
| `a_star.py` | Classe `AStar` e busca em largura de referência |
| `robot.py` | Agente guloso da Atividade 1 |
| `maze.py` | Ambiente e regras de movimentação |
| `read_matrix.py` | Leitura do arquivo de entrada |
| `visualization.py` | Mapas comparativos e gráfico de métricas |
| `data/` | Labirintos de teste |
| `resultados/` | Figuras e CSV gerados |


## Pré-requisitos

```bash
pip install numpy matplotlib
```

## Como Rodar

```bash
python main.py data/IA_2_labirinto256.txt
```

Sem argumento, o programa pergunta o caminho do arquivo. As saídas são gravadas
em `resultados/`: `comparacao_gulosa_vs_astar.png`, `grafico_metricas.png`,
`astar_por_metrica.png` e `metricas.csv`.

O programa aceita os dois formatos de entrada: o da Atividade 1, com o objetivo
marcado por um valor negativo dentro da matriz, e o do Desafio, com as
coordenadas do objetivo na última linha do arquivo.

## As Duas Técnicas Comparadas

| | Agente guloso | Busca informada A\* |
|---|---|---|
| Natureza | Agente situado, ocupa uma posição | Algoritmo sobre o grafo completo |
| Percepção | Apenas os 8 vizinhos imediatos | Mapa inteiro |
| Decisão | Vizinho mais próximo do objetivo | Menor `f = g + h` na fronteira |
| Becos sem saída | Recua fisicamente, gastando laços | Salta para outra região sem custo |
| Garantia | Nenhuma | Ótimo, se a heurística for admissível |

O agente decide olhando só o próximo passo e não reconsidera. O A\* soma o custo
já percorrido (`g`) à estimativa do que falta (`h`), de forma que um caminho que
parecia bom mas com custo alto vai para o fim da fila.



## Modelo de custo e admissibilidade

Seguindo a definição de *laço de execução* da Atividade 1, **todo movimento
custa 1**, inclusive o diagonal, na vizinhança de 8 conexões. Sob esse modelo, o
custo real restante em terreno livre é exatamente a **distância de Chebyshev**.

Uma heurística é admissível quando nunca superestima o custo restante, é essa
propriedade que garante que o A\* encontre o caminho ótimo.

| Métrica | Admissível? | Por quê |
|---|---|---|
| **Chebyshev** | **Sim** | Iguala o custo real: nunca superestima |
| Euclidiana | Não | Mede a diagonal como √2 ≈ 1,41, mas ela custa 1 |
| Manhattan | Não | Cobra 2 por um passo diagonal que custa 1 |
| Nula | Sim | Não estima nada; reduz o A\* a Dijkstra |

As métricas inadmissíveis foram mantidas propositalmente para 
demonstrar experimentalmente a perda da garantia de otimalidade.


## Resultados (matriz 256×256, objetivo em 255,255)

| Técnica | Métrica | Passos | Posições exploradas | Laços |
|---|---|---|---|---|
| Gulosa | Euclidiana | 326 | 327 | 326 |
| Gulosa | Manhattan | 322 | 327 | 330 |
| Gulosa | Chebyshev | 392 | 394 | 394 |
| A\* | Euclidiana | 304 | 436 | 304 |
| A\* | Manhattan | 310 | 347 | 310 |
| **A\*** | **Chebyshev** | **291** | **8.060** | 291 |
| A\* | Nula (Dijkstra) | 291 | 45.739 | 291 |
| BFS | — | 291 | 45.739 | 291 |

### Conclusões
O A\* com Chebyshev atinge o ótimo de 291 passos explorando 8.060 posições,
**82,4% menos** que as 45.739 da busca cega. 
O melhor agente guloso fica 35 passos acima do ótimo (+12,0%), mas explora
apenas 327 posições. Ou seja o agente reativo é barato, mas enxerga mal
e o algoritmo informado é caro e ótimo.

A Chebyshev foi a pior métrica no agente guloso  com 392 passos, com 114 decisões
empatadas e a única correta no A\*. O custo acumulado `g` neutraliza
exatamente os empates que a prejudicavam os resultados iniciais da Atividade 1.

Pelo número de laços vence a Euclidiana com 326 contra 330. Já pelo comprimento do
caminho final vence a Manhattan  com 322 contra 326. A diferença são os 4 recuos da
Manhattan, porque ela entra em quatro becos e gasta mais laços para sair, mas a rota
que resta depois de descartá-los é mais curta. Como o enunciado da Atividade 1 define 
o desempenho pelo número de laços, esse o critério  foi mantido no código.
