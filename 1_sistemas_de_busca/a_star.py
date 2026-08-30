import math
import time
import heapq
from typing import Any, Dict, List, Tuple, Optional
from collections import deque


class AStar:
    """
        Busca o caminho de menor custo usando o algoritmo A*.
        A definição de laço de execução da Atividade 1 foi mantida, então todo movimento
        custa 1, inclusive o diagonal.
    """
    NEIGHBOR_OFFSETS: List[Tuple[int, int]] = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    def __init__(self, maze: Any, start: Tuple[int, int] = (0, 0), metric: str = 'chebyshev'):
        """
            Inicializa a busca A*.
        
            Args:
                maze (Maze): A instância do labirinto. A matriz não é modificada.
                start (Tuple[int, int], optional): Coordenada inicial. Padrão é (0, 0).
                metric (str, optional): Métrica usada como heurística. Padrão é 'chebyshev'.
        """
        self.maze = maze
        self.start = start
        self.metric = metric

        self.path: List[Tuple[int, int]] = []
        self.expanded: List[Tuple[int, int]] = []
        self.cost: float = 0.0
        self.elapsed: float = 0.0

    def heuristic(self, position: Tuple[int, int]) -> float:
        """
            Estima o custo da posição informada até o objetivo.

        Args:
            position (Tuple[int, int]): Coordenada a ser avaliada.

        Returns:
            float: A distância estimada até o objetivo.
        """
        row, col = position
        goal_row, goal_col = self.maze.goal_position
        delta_row = abs(row - goal_row)
        delta_col = abs(col - goal_col)

        if self.metric == 'chebyshev':
            return float(max(delta_row, delta_col))
        elif self.metric == 'manhattan':
            return float(delta_row + delta_col)
        elif self.metric == 'euclidean':
            return math.hypot(delta_row, delta_col)
        elif self.metric == 'zero':
            return 0.0
        else:
            return float(max(delta_row, delta_col))


    def is_admissible(self) -> bool:
        """
            Informa se a métrica escolhida preserva a garantia de otimalidade.

            Sob custo unitário, apenas a distância de Chebyshev e a heurística nula
            nunca superestimam o custo restante.

            Returns:
                bool: True se a métrica for admissível.
        """
        return self.metric in ('chebyshev', 'zero')


    def build_path(self, came_from: Dict, goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
            Reconstrói o caminho da origem até o objetivo, percorrendo os antecessores.
        
            Args:
                came_from (Dict): Mapeamento de cada posição para a posição anterior.
                goal (Tuple[int, int]): Coordenada final.
        
            Returns:
                List[Tuple[int, int]]: Sequência de coordenadas da origem ao objetivo.
        """
        path = [goal]
        while path[-1] in came_from:
            path.append(came_from[path[-1]])
        path.reverse()

        return path

    def run(self) -> bool:
        """
            Executa a busca A* sobre o labirinto.
        
            Returns:
                bool: True se o objetivo foi alcançado, False caso contrário.
        """
        start_time = time.perf_counter()
        goal = self.maze.goal_position

        frontier = [(self.heuristic(self.start), self.heuristic(self.start), self.start)]
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score = {self.start: 0.0}
        closed = set()
        success = False

        while frontier:
            f, h, current = heapq.heappop(frontier)

            if current in closed:
                continue
            closed.add(current)
            self.expanded.append(current)

            if current == goal:
                success = True
                break

            row, col = current
            for delta_row, delta_col in self.NEIGHBOR_OFFSETS:
                neighbor = (row + delta_row, col + delta_col)

                if not self.maze.is_walkable(*neighbor) or neighbor in closed:
                    continue

                tentative_g = g_score[current] + 1.0

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h = self.heuristic(neighbor)
                    heapq.heappush(frontier, (tentative_g + h, h, neighbor))

        self.elapsed = time.perf_counter() - start_time

        if success:
            self.path = self.build_path(came_from, goal)
            self.cost = g_score[goal]

        return success


def breadth_first_search(maze: Any, start: Tuple[int, int] = (0, 0)) -> dict:
    """
    Busca em largura usada como referência na comparação.

    Args:
        maze (Maze): A instância do labirinto.
        start (Tuple[int, int], optional): Coordenada inicial. Padrão é (0, 0).

    Returns:
        dict: Resultado no mesmo formato usado pelas outras técnicas.
    """
    start_time = time.perf_counter()
    goal = maze.goal_position

    queue = deque([start])
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    visited = {start}
    expanded: List[Tuple[int, int]] = []
    success = False

    while queue:
        current = queue.popleft()
        expanded.append(current)

        if current == goal:
            success = True
            break

        row, col = current
        for delta_row, delta_col in AStar.NEIGHBOR_OFFSETS:
            neighbor = (row + delta_row, col + delta_col)
            if neighbor not in visited and maze.is_walkable(*neighbor):
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)

    elapsed = time.perf_counter() - start_time

    path = []
    if success:
        path = [goal]
        while path[-1] in came_from:
            path.append(came_from[path[-1]])
        path.reverse()

    return {
        'technique': 'BFS',
        'metric': 'nenhuma',
        'success': success,
        'path': path,
        'explored': expanded,
        'steps': max(len(path) - 1, 0),
        'cost': float(max(len(path) - 1, 0)),
        'expanded': len(expanded),
        'loops': max(len(path) - 1, 0),
        'time': elapsed,
        'warning': '',
    }


    