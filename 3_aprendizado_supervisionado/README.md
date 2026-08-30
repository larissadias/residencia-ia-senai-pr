# Projeto Final Parte 3: Aprendizado Supervisionado aplicado à Concessão de Crédito
**Disciplina:** Inteligência Artificial
**Professor:** Prof. Rafael Bressan
**Aluna:** Ana Larissa Dias


Comparação de modelos de classificação sobre o dataset de concessão de crédito,
com a escolha dos algoritmos fundamentada em revisões sistemáticas da literatura
de *credit scoring*.

Todas as sementes de aleatoriedade estão fixadas em **26**, para que os
resultados sejam reprodutíveis.

## Estrutura dos arquivos

| Arquivo | Papel |
|---|---|
| `split_data.py` | Gera e valida a divisão treino/teste |
| `load_data.py` | Carrega os conjuntos e trata os códigos de ausência |
| `metrics.py` | Funções de avaliação usadas nas duas fases |
| `plots.py` | Figuras usadas como evidência visual |
| `run_baseline.py` | Fase 1: compara os modelos |
| `run_refinements.py` | Fase 2: refina o vencedor |
| `treino_ids.txt`, `teste_ids.txt` | Registro da divisão |

As funções de avaliação ficam em módulo separado para que as duas fases usem
exatamente o mesmo cálculo.


## Pré-requisitos

```bash
pip install numpy pandas scikit-learn matplotlib xgboost
```

O arquivo `dataset.csv` deve estar em `data/`.

## Pipeline de Execução

```bash
python split_data.py        # gera treino_ids.txt e teste_ids.txt
python load_data.py         # verifica a carga e os valores ausentes
python run_baseline.py      # fase 1: compara os modelos
python run_refinements.py   # fase 2: refina o vencedor
```

A divisão treino/teste é gerada uma única vez. Rodar `split_data.py`
novamente sorteia outros conjuntos e invalida a comparação com os resultados
registrados. Por isso, os arquivos teste_ids.txt e treino.ids.txt estão 
no mesmo diretório.


## Dataset de Concessão de Crédito

| | |
|---|---|
| Registros | 219.984 |
| Variáveis | 672, anonimizadas (`v1` … `v706`) |
| Alvo | `Y`, binário |
| Classe 1 | 79,52% |
| Classe 0 | 20,48% |
| Campos vazios | nenhum |

As variáveis não têm significado documentado, por isso não se sabe se representam dados
pessoais ou movimentação bancária e nem o que a variável-alvo codifica, bom ou mal pagador.


## Levantamento Bibliográfico

Sem saber o que as variáveis representam, optou-se por realizar um levantamento
bibliográfico na área de credit scoring, levantando quais técnicas os trabalhos 
publicados mais empregam e quais apresentam melhor desempenho nesse tipo de problema.

Foram consultadas duas revisões sistemáticas:

**Dastile, Celik e Potsane (2020)**, *Applied Soft Computing*  com 74 estudos
publicados entre 2010 e 2018, com meta-análise de frequência de uso, métricas e
desempenho nos datasets German e Australian.

**Ayari, Guetari e Kraïem (2026)**, *Artificial Intelligence Review* com  63 estudos
publicados entre 2018 e 2024, cobrindo o período mais recente.

O levantamento mostrou que:

**Ensembles superam classificadores individuais.** É a conclusão central de
Dastile et al., derivada da meta-análise dos 74 estudos.

**Regressão logística e árvore de decisão como referências.** O framework
proposto por Dastile et al. recomenda esses dois como *benchmarks*: a regressão
logística por ser o padrão histórico do setor e ter desempenho semelhante ao de
muitos modelos tradicionais, e a árvore de decisão pela capacidade de explicar
suas decisões.

**Ensembles de árvore lidam bem com desbalanceamento.** Brown e Mues (2012),
citados por Dastile et al., mostraram que random forest e gradient boosting se
saem bem em contextos de credit scoring e toleram desbalanceamento acentuado,
enquanto o KNN tem desempenho significativamente pior sob desbalanceamento forte.

**KNN é computacionalmente caro.** Henley e Hand (1996), também citados na
revisão, apontam que o método exige calcular uma distância métrica para cada
registro armazenado no momento da classificação.

**Acurácia e AUC são as métricas mais reportadas**, com 48 e 30 ocorrências no
primeiro período e 49 e 31 no segundo. As duas revisões alertam, porém, que a
acurácia é enganosa sob desbalanceamento, o que sustenta a escolha da AUC como
métrica principal.

**XGBoost é reconhecido por velocidade e desempenho**, e no período recente o
gradient boosting sobre árvores consolidou-se como estado da arte prático em
dados tabulares.

O levantamento completo está em `levantamento-bibliografico-credit-scoring.docx`,
anexo a esta entrega.

### Modelos selecionados

| Modelo | Papel na comparação |
|---|---|
| Baseline (classe majoritária) | o que se obtém sem olhar variável alguma |
| Regressão Logística | *Benchmark* recomendado pela literatura |
| Árvore de Decisão | *Benchmark* interpretável recomendado pela literatura |
| Random Forest | Ensemble com bom desempenho sob desbalanceamento |
| XGBoost | Estado da arte prático em dados tabulares |
| KNN | Incluído para testar a previsão da literatura |



## Tratamento dos valores ausentes

Uma verificação com `isna()` retorna zero faltantes. O livro Python for Data Analysis
alertou que valores como `-999` podem ser códigos usados para marcar dados
ausentes. Inspecionando os valores negativos da base, encontraram-se seis códigos 
que destoam do restante dos dados:
`-997`, `-998`, `-999`, `-9997`, `-9998` e `-9999`. 
Eles aparecem isolados em colunas que os valores válidos vão de 0 a 5. Além disso, 
eles repetem demais. O valor `-9999` chega a ocupar 92,3% da coluna `v633`.
A base contém valores negativos de `-1` a `-55`.Por isso, criou-se o MISSING_THRESHOLD = -900,
o limiar de −999 deixaria passar `-997` e `-998`, que são maiores que ele.

Entendeu-se que tratá-los como outliers não seria adequado, considerando que um outlier é uma observação atípica, e um valor presente em 92% dos registros é a norma daquela coluna.

**Resultado:** 5,44% das células do treino são ausentes, em 44 das 672 colunas.
Dessas, 43 passam de 50% de ausência e 17 passam de 90%.

## Experimento
### Divisão treino e teste

Divisão de 80/20 estratificada, gravada como listas de identificadores em
`treino_ids.txt` e `teste_ids.txt`.

| | Registros | Classe 1 |
|---|---|---|
| Treino | 175.987 | 79,52% |
| Teste | 43.997 | 79,51% |

Verificações executadas: nenhum registro em comum entre os conjuntos, e a
divisão cobre o dataset inteiro.

A escolha de gravar os identificadores em arquivo, em vez de copiar os dados, resolveu dois
problemas. Com 219.984 registros e 674 colunas, duplicar o arquivo ocuparia mais
de um gigabyte. E registrando quais amostras compõem cada conjunto garante a reprodutibilidade
do experiemnto.

O conjunto de teste não recebe nenhum tratamento derivado dele próprio.
A mediana usada para preencher os ausentes e a média e o desvio usados para
padronizar vêm apenas do treino, e são então aplicados ao teste.

O mesmo split é usado nos seis modelos, sem novo sorteio entre eles. Sem isso a
comparação não seria justa.

Cada modelo com componente aleatório é executado três vezes, com sementes
diferentes e o mesmo split, e o resultado é reportado com média e desvio.
Modelos determinísticos rodam uma vez, e isso é registrado na tabela.

### Métrica Principal

Um modelo que responda sempre a classe majoritária, sem examinar nenhuma
variável, alcança **79,51% de acurácia**. Por isso a métrica principal é a
**ROC AUC**, que mede a capacidade de ordenar os casos por risco
independentemente do limiar de decisão. É também, junto da acurácia, a métrica
mais usada na literatura da área.

## Fase 1 — Experimento base

### Qualidade

| Modelo | AUC | Acurácia | Acur.Bal | F1(1) | Recall(1) | Recall(0) |
|---|---|---|---|---|---|---|
| Baseline (classe majoritária) | 0,5000 | 0,7951 | 0,5000 | 0,8859 | 1,0000 | 0,0000 |
| Regressão Logística | 0,6798 | 0,8013 | 0,5426 | 0,8870 | 0,9808 | 0,1043 |
| Árvore de Decisão | 0,6782 ± 0,0001 | 0,8030 | 0,5553 | 0,8873 | 0,9750 | 0,1355 |
| Random Forest | 0,6968 ± 0,0008 | 0,8040 | 0,5393 | 0,8890 | 0,9876 | 0,0911 |
| KNN | 0,5954 | 0,7967 | 0,5369 | 0,8843 | 0,9771 | 0,0966 |
| **XGBoost** | **0,7204 ± 0,0000** | 0,8073 | 0,5649 | 0,8895 | 0,9756 | 0,1541 |

### Custo computacional

| Modelo | Treino | ms por amostra | Execuções |
|---|---|---|---|
| Regressão Logística | 13,0s | 0,001 | 1 (determinístico) |
| Árvore de Decisão | 8,1s | 0,000 | 3 sementes |
| Random Forest | 6,9s | 0,009 | 3 sementes |
| KNN | 0,0s | **0,202** | 1 (determinístico) |
| XGBoost | 11,8s | 0,001 | 3 sementes |

O tempo para classificar uma amostra é reportado porque o custo de um modelo não
está apenas no treino. Sem essa coluna o KNN pareceria o mais barato da tabela:
ele treina em zero segundo porque não treina de verdade, guarda o conjunto e
adia todo o cálculo para a predição, como a literatura registra.

### Conclusões

**Em acurácia todos os modelos parecem equivalentes**, de 0,795 a 0,807, colados
na linha do chute. A diferença só aparece na AUC.

**As previsões da literatura se confirmaram.** O XGBoost venceu, e o KNN obteve a
pior AUC de todas (0,5954), abaixo até da regressão logística, além de ser
duzentas vezes mais lento que os demais na classificação. Os dois resultados
estavam antecipados nos trabalhos consultados.

**A variação entre sementes é desprezível**  Árvore ±0,0001, Random Forest
±0,0008. Com 175.987 registros de treino os modelos ficam bem determinados, de
modo que as diferenças de AUC entre eles são reais e não ruído de sorteio.


## Fase 2 — Refinamento

O vencedor é mantido fixo e as intervenções são feitas uma de cada vez,
comparadas contra o mesmo ponto de partida. Não se re-executam todos os modelos
a cada mudança.

### Ajuste de hiperparâmetros

| Configuração | AUC | vs base | Acur.Bal | Recall(0) | Treino |
|---|---|---|---|---|---|
| Baseline (100 árvores) | 0,7204 | — | 0,5649 | 0,1541 | 11,8s |
| **Mais árvores (300)** | **0,7210** | +0,0006 | 0,5686 | 0,1640 | 23,0s |
| Árvores mais profundas (10) | 0,7175 | −0,0029 | 0,5687 | 0,1662 | 20,8s |
| Amostragem 0,8 | 0,7206 | +0,0002 | 0,5648 | 0,1538 | 11,9s |
| Peso por classe | 0,7197 | −0,0007 | **0,6582** | **0,5828** | 11,9s |

**Nenhum hiperparâmetro move a AUC de forma relevante.** A maior variação entre
as quatro configurações foi 0,0035.

Vale registrar o custo da configuração vencedora: **dobrar o tempo de treino
comprou 0,0006 de AUC**. Em uso prático, o modelo de 100 árvores seria a escolha.

O "Peso por classe" merece atenção separada, pois ele **piorou** a AUC e mesmo assim
elevou a acurácia balanceada em 9,3 pontos e quase quadruplicou o recall da
classe 0.

### Ajuste do limiar de decisão

O limiar padrão de 0,5 pressupõe classes equilibradas. Com quase 80% dos
registros em uma única classe, ele leva o modelo a responder a classe majoritária
quase sempre.

Esta intervenção responde a uma lacuna que a própria revisão de Dastile et al.
aponta: entre as limitações da literatura de credit scoring, os autores registram
que o uso de pontos de corte diferentes de 0,5 para classificar tomadores **não é
coberto** pelos trabalhos analisados.

O limiar foi definido pelo **índice de Youden** (Youden, 1950), que maximiza a
diferença entre a taxa de verdadeiros positivos e a de falsos positivos.

| | Limiar 0,50 | Limiar 0,81 |
|---|---|---|
| AUC | 0,7210 | **0,7210** |
| Acurácia | 0,8074 | 0,6628 |
| Acurácia balanceada | 0,5686 | **0,6601** |
| Recall da classe 0 | 0,1640 | **0,6556** |
| F1 da classe 1 | 0,8894 | 0,7582 |

**Matriz de confusão, limiar 0,50**

| | previsto 0 | previsto 1 |
|---|---|---|
| real 0 | 1.478 (16,4%) | 7.535 |
| real 1 | 937 | 34.047 (97,3%) |

**Matriz de confusão, limiar 0,81**

| | previsto 0 | previsto 1 |
|---|---|---|
| real 0 | **5.909 (65,6%)** | 3.104 |
| real 1 | 11.731 | 23.253 (66,5%) |

**A AUC não muda com o limiar**, porque ela mede a ordenação e não a decisão. A
acurácia **cai** de 0,8074 para 0,6628, e ainda assim o modelo se torna mais
útil: a identificação da classe minoritária passa de 16,4% para 65,6%.

## Pipeline final

1. **Divisão**: 80/20 estratificada, com os identificadores registrados em
   arquivo. Conjunto de teste congelado.
2. **Valores ausentes**: converter os valores menores ou iguais a −900
3. **Codificação e escala**: nenhuma. O XGBoost trata ausentes nativamente e
   divide por limiares, o que dispensa imputação e padronização e remove duas
   escolhas arbitrárias do pipeline.
4. **Modelo**: XGBoost com 300 árvores, profundidade 6, taxa de aprendizado 0,1.
5. **Decisão**: limiar de 0,81, definido pelo índice de Youden.

**Resultado:** AUC 0,7210, acurácia balanceada 0,6601, identifica 65,6% da classe
minoritária. Treina em 23 segundos e classifica uma amostra em 0,001 ms.


## Figuras

| Arquivo | Gerado por | O que mostra |
|---|---|---|
| `results/figura_comparacao_modelos.png` | `run_baseline.py` | Acurácia e AUC lado a lado: por que a primeira não serve |
| `results/figura_limiar.png` | `run_refinements.py` | Curva ROC e as duas matrizes de confusão |
| `results/figura_analise_erros.png` | `run_refinements.py` | Sobreposição das probabilidades das duas classes |



## Referências

DASTILE, X.; CELIK, T.; POTSANE, M. Statistical and machine learning models in
credit scoring: a systematic literature survey. 2020.

AYARI, H.; GUETARI, R.; KRAÏEM, N. Machine learning powered financial credit
scoring: a systematic literature review. 2026.

McKinney, Wes. Python for Data Analysis: Data Wrangling with pandas, NumPy, 
and Jupyter. 3rd ed., O'Reilly Media, 2022.
