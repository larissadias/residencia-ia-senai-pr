import os
import sys
import copy
import csv
import time
from typing import List, Tuple, Optional
from read_matrix import read_matrix
from maze import Maze
from robot import Robot
from a_star import AStar, breadth_first_search
from visualization import generate_comparison_map, generate_metrics_chart, open_images

# Métricas testadas pelo agente guloso da Atividade 1
METRICAS = ['euclidean', 'manhattan', 'chebyshev']

NOMES = {
    'euclidean': 'Euclidiana',
    'manhattan': 'Manhattan',
    'chebyshev': 'Chebyshev',
    'zero': 'Nula',
    'nenhuma': 'nenhuma',
}

PASTA_SAIDA = 'resultados'


def obter_caminho_arquivo() -> str:
    """
        Lê o arquivo passado via argumento no terminal ou solicita input do usuário.

        Returns:
            str: O caminho relativo ou absoluto para o arquivo de texto.
    """
    if len(sys.argv) > 1:
        return sys.argv[1]

    print("\nErro: Nenhum arquivo de entrada foi informado no comando inicial.")
    while True:
        caminho = input("Por favor, digite o caminho do arquivo (ex: data/IA_2_labirinto256.txt): ").strip()
        if caminho:
            return caminho
        print("Caminho vazio! Você precisa digitar o local do arquivo.")


def novo_labirinto(matriz: List[List[int]], objetivo: Optional[Tuple[int, int]]) -> Maze:
    """
        Cria uma instância limpa do labirinto a partir da matriz original.

        Args:
            matriz (List[List[int]]): A matriz original do labirinto.
            objetivo (Optional[Tuple[int, int]]): Coordenadas do objetivo, se o
            arquivo as informar na última linha.

        Returns:
            Maze: Uma nova instância independente do ambiente.
    """
    return Maze(copy.deepcopy(matriz), goal_position=objetivo)


def executar_guloso(matriz: List[List[int]], objetivo: Optional[Tuple[int, int]],
                    metrica: str) -> dict:
    """
        Executa o agente reativo da Atividade 1 e organiza suas métricas.

        Args:
            matriz (List[List[int]]): A matriz original do labirinto.
            objetivo (Optional[Tuple[int, int]]): Coordenadas do objetivo.
            metrica (str): A métrica de distância usada pelo agente.

        Returns:
            dict: Resultado da execução, com o caminho final já separado da trilha.
    """
    maze = novo_labirinto(matriz, objetivo)
    robot = Robot(maze, start=(0, 0), metric=metrica)
    maze.mark_visited(0, 0)

    inicio = time.perf_counter()
    try:
        sucesso = robot.run()
        aviso = ''
    except RuntimeError as e:
        sucesso = False
        aviso = str(e)
    tempo = time.perf_counter() - inicio

    caminho = robot.final_path() if sucesso else []
    empates = sum(1 for c in robot.tie_counts if c > 1)

    return {
        'technique': 'Gulosa',
        'metric': NOMES[metrica],
        'success': sucesso,
        'path': caminho,
        'explored': robot.path,
        'goal': maze.goal_position,
        'steps': max(len(caminho) - 1, 0),
        'cost': float(max(len(caminho) - 1, 0)),
        'expanded': robot.visited_cells(),
        'loops': robot.loop_count,
        'ties': empates,
        'time': tempo,
        'warning': aviso,
    }


def executar_astar(matriz: List[List[int]], objetivo: Optional[Tuple[int, int]],
                   metrica: str) -> dict:
    """
        Executa a busca A* e organiza suas métricas.
    
        Args:
            matriz (List[List[int]]): A matriz original do labirinto.
            objetivo (Optional[Tuple[int, int]]): Coordenadas do objetivo.
            metrica (str): A métrica usada como heurística.

        Returns:
            dict: Resultado da execução do A*.
    """
    maze = novo_labirinto(matriz, objetivo)
    planejador = AStar(maze, start=(0, 0), metric=metrica)
    sucesso = planejador.run()

    aviso = ''
    if not planejador.is_admissible():
        aviso = (f"A métrica {NOMES[metrica]} superestima o custo restante e por isso "
                 f"é inadmissível: o caminho pode não ser o ótimo.")

    return {
        'technique': 'A*',
        'metric': NOMES[metrica],
        'success': sucesso,
        'path': planejador.path,
        'explored': planejador.expanded,
        'goal': maze.goal_position,
        'steps': max(len(planejador.path) - 1, 0),
        'cost': planejador.cost,
        'expanded': len(planejador.expanded),
        'loops': max(len(planejador.path) - 1, 0),
        'ties': 0,
        'time': planejador.elapsed,
        'warning': aviso,
    }


def imprimir_tabela(resultados: List[dict]) -> None:
    """
        Imprime no terminal a tabela comparativa das técnicas executadas.

        Args:
            resultados (List[dict]): Os resultados a serem exibidos.
    """
    cabecalho = f"{'Técnica':<8} {'Métrica':<12} {'Passos':>7} {'Explorado':>10} {'Laços':>7} {'Tempo (s)':>10}"
    print(cabecalho)
    print("-" * len(cabecalho))
    for r in resultados:
        passos = r['steps'] if r['success'] else '-'
        print(f"{r['technique']:<8} {r['metric']:<12} {passos:>7} "
              f"{r['expanded']:>10} {r['loops']:>7} {r['time']:>10.4f}")


def salvar_csv(resultados: List[dict], caminho_saida: str) -> str:
    """
        Grava a tabela comparativa em um arquivo CSV.

        Args:
            resultados (List[dict]): Os resultados a serem gravados.
            caminho_saida (str): Caminho do arquivo CSV.

        Returns:
            str: O caminho do arquivo salvo.
    """
    os.makedirs(os.path.dirname(os.path.abspath(caminho_saida)), exist_ok=True)
    colunas = ['technique', 'metric', 'success', 'steps', 'cost',
               'expanded', 'loops', 'time']

    with open(caminho_saida, 'w', newline='', encoding='utf-8') as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(colunas)
        for r in resultados:
            escritor.writerow([r[c] for c in colunas])

    return caminho_saida


def main():
    """
        Compara o agente guloso da Atividade 1 com o algoritmo A* no mesmo labirinto.
    """
    file_path = obter_caminho_arquivo()

    if not os.path.exists(file_path):
        print(f"Erro: O arquivo '{file_path}' não foi encontrado.")
        return

    print(f"Lendo a matriz do arquivo: {file_path}...")
    matriz, objetivo = read_matrix(file_path)

    referencia = novo_labirinto(matriz, objetivo)
    print(f"Matriz {referencia.num_rows}x{referencia.num_cols} | "
          f"início (0, 0) | objetivo {referencia.goal_position}\n")

    # Etapa 1: agente guloso nas três métricas
    print("Executando o agente guloso da Atividade 1...")
    gulosos = []
    for metrica in METRICAS:
        r = executar_guloso(matriz, objetivo, metrica)
        gulosos.append(r)
        estado = f"{r['steps']} passos" if r['success'] else "sem solução"
        print(f"  - {r['metric']:<12} {estado}, {r['loops']} laços, "
              f"{r['ties']} empates")

    validos = [r for r in gulosos if r['success']]
    melhor_guloso = min(validos, key=lambda r: (r['loops'], r['steps'])) if validos else gulosos[0]
    print(f"  -> melhor métrica gulosa (menor número de laços): {melhor_guloso['metric']}\n")

    # A* nas três métricas
    print("Executando o algoritmo A*...")
    astares = []
    for metrica in METRICAS:
        r = executar_astar(matriz, objetivo, metrica)
        astares.append(r)
        estado = f"{r['steps']} passos" if r['success'] else "sem solução"
        print(f"  - {r['metric']:<12} {estado}, {r['expanded']} posições exploradas")

    astar_principal = [r for r in astares if r['metric'] == 'Chebyshev'][0]
    print()

    # Busca em largura para comparação 
    print("Executando a busca em largura (piso teórico)...")
    bfs = breadth_first_search(novo_labirinto(matriz, objetivo))
    bfs['goal'] = referencia.goal_position
    bfs['metric'] = 'nenhuma'
    bfs['ties'] = 0
    estado = f"{bfs['steps']} passos" if bfs['success'] else "sem solução"
    print(f"  - BFS: {estado}, {bfs['expanded']} posições exploradas\n")

    print("Validando a implementação do A* (heurística nula = Dijkstra)...")
    dijkstra = executar_astar(matriz, objetivo, 'zero')

    if dijkstra['steps'] != bfs['steps']:
        print(f"  - ATENÇÃO: o A* sem heurística encontrou {dijkstra['steps']} passos "
              f"e a busca em largura {bfs['steps']}. Há um erro na implementação.\n")
    else:
        diferenca = abs(dijkstra['expanded'] - bfs['expanded'])
        print(f"  - OK: mesmo caminho de {bfs['steps']} passos que a busca em largura.")
        if diferenca == 0:
            print(f"  - As duas técnicas exploraram as mesmas {bfs['expanded']} posições.\n")
        else:
            print(f"  - Posições exploradas: {dijkstra['expanded']} contra "
                  f"{bfs['expanded']} da busca em largura (diferença de {diferenca}).")
            print(f"    A diferença vem apenas da ordem de desempate na última "
                  f"camada de posições equidistantes do objetivo.\n")

    resultados = gulosos + astares + [dijkstra, bfs]

    # Tabela comparativa
    print("=" * 60)
    print("TABELA COMPARATIVA")
    print("=" * 60)
    imprimir_tabela(resultados)
    print()

    if astar_principal['success'] and bfs['success']:
        economia = 100 * (1 - astar_principal['expanded'] / bfs['expanded'])
        if astar_principal['steps'] == bfs['steps']:
            print(f"O A* com Chebyshev atingiu o ótimo de {bfs['steps']} passos, "
                  f"explorando {astar_principal['expanded']} posições contra "
                  f"{bfs['expanded']} da busca cega ({economia:.1f}% menos).")

    if melhor_guloso['success'] and bfs['success']:
        excesso = melhor_guloso['steps'] - bfs['steps']
        pct = 100 * excesso / bfs['steps']
        print(f"O melhor agente guloso ({melhor_guloso['metric']}) ficou {excesso} passos "
              f"acima do ótimo (+{pct:.1f}%), explorando apenas "
              f"{melhor_guloso['expanded']} posições.")

    for r in astares:
        if r['warning'] and r['success']:
            print(f"[!] A* ({r['metric']}): {r['steps']} passos. {r['warning']}")
    print()

    # Figuras e planilha de métricas
    print("Gerando figuras...")

    mapa = generate_comparison_map(
        matriz, [melhor_guloso, astar_principal],
        os.path.join(PASTA_SAIDA, 'comparacao_gulosa_vs_astar.png'),
        'Agente Reativo com Heurística Gulosa  x  Algoritmo de Busca Informada A*')
    grafico = generate_metrics_chart(
        [melhor_guloso, astar_principal, bfs],
        os.path.join(PASTA_SAIDA, 'grafico_metricas.png'))
    heuristicas = generate_comparison_map(
        matriz, astares,
        os.path.join(PASTA_SAIDA, 'astar_por_metrica.png'),
        'Efeito da heurística sobre o A* (custo diagonal = 1)')
    planilha = salvar_csv(resultados, os.path.join(PASTA_SAIDA, 'metricas.csv'))

    for arquivo in (mapa, grafico, heuristicas, planilha):
        print(f"  - {arquivo}")

    open_images([mapa, grafico])


if __name__ == "__main__":
    main()
