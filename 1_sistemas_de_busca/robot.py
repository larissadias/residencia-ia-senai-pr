import math
from typing import List, Tuple, Optional, Any

class Robot:
    """
    Agente inteligente que navega pelo labirinto representado por um objeto Maze.

    A cada passo, o agente segue o ciclo clássico de Percepção -> Decisão -> Ação:
      - Percepção: Observa as 8 posições adjacentes e filtra aquelas por onde pode caminhar.
      - Decisão: Escolhe avançar para o vizinho com a menor distância até o objetivo, 
                 ou recuar (backtrack) se não houver vizinho válido.
      - Ação: Executa a decisão, atualizando a matriz, a pilha de posições visitadas 
              e o contador de laços.
    """

    # Deslocamentos (delta_linha, delta_coluna) para as 8 posições adjacentes 
    # Vizinhança de conectividade-8
    NEIGHBOR_OFFSETS: List[Tuple[int, int]] = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    def __init__(self, maze: Any, start: Tuple[int, int] = (0, 0), metric: str = 'euclidean'):
        """
        Inicializa o agente e suas estruturas de memória.

        Args:
            maze (Maze): A instância do labirinto onde o agente irá navegar.
            start (Tuple[int, int], optional): Coordenadas iniciais do agente. Padrão é (0, 0).
            metric (str, optional): A métrica de distância a ser utilizada. Padrão é 'euclidean'.
        """
        self.maze = maze
        self.position = start
        
        # Pilha LIFO (último a entrar, primeiro a sair) das posições percorridas anteriormente
        self.stack: List[Tuple[int, int]] = []       
        
        # Contador de laços de execução
        self.loop_count: int = 0   
        
        # Histórico completo de posições, útil para testes e geração do mapa final
        self.path: List[Tuple[int, int]] = [start]   
        
        # Armazena a métrica escolhida pelo usuário
        self.metric: str = metric  
        
        # Lista para registrar empates a cada decisão
        self.tie_counts: List[int] = []  
    
    
    # ------------------------------------------------------------------
    # Percepção
    # ------------------------------------------------------------------
    def perceive(self) -> List[Tuple[int, int]]:
        """
        Observa o ambiente ao redor do agente.
        
        Verifica as 8 posições adjacentes à atual e retorna apenas as coordenadas
        transitáveis (valor 1) ou aquelas que representam o objetivo (valor -1).

        Returns:
            List[Tuple[int, int]]: Lista de coordenadas (linha, coluna) válidas para movimento.
        """
        row, col = self.position
        valid_neighbors: List[Tuple[int, int]] = []

        for delta_row, delta_col in self.NEIGHBOR_OFFSETS:
            neighbor = (row + delta_row, col + delta_col)
            if self.maze.is_walkable(*neighbor):
                valid_neighbors.append(neighbor)

        return valid_neighbors
    

    # ------------------------------------------------------------------
    # Decisão
    # ------------------------------------------------------------------
    @staticmethod
    def _euclidean_distance(position_a: Tuple[int, int], position_b: Tuple[int, int]) -> float:
        row_a, col_a = position_a
        row_b, col_b = position_b
        return math.hypot(row_a - row_b, col_a - col_b)
    
    @staticmethod
    def _minkowski_distance(position_a: Tuple[int, int], position_b: Tuple[int, int], p: int) -> float:
        row_a, col_a = position_a
        row_b, col_b = position_b 
        return (abs(row_a - row_b)**p + abs(col_a - col_b)**p) ** (1/p) 
    
    @staticmethod
    def _chebyshev_distance(position_a: Tuple[int, int], position_b: Tuple[int, int]) -> float:
        row_a, col_a = position_a
        row_b, col_b = position_b
        return max(abs(row_a - row_b), abs(col_a - col_b))

    def decide(self, valid_neighbors: List[Tuple[int, int]]) -> Tuple[str, Optional[Tuple[int, int]]]:
        """
        Determina o próximo passo com base na percepção e na métrica escolhida.

        Args:
            valid_neighbors (List[Tuple[int, int]]): Lista de posições vizinhas disponíveis.

        Returns:
            Tuple[str, Optional[Tuple[int, int]]]: Uma tupla contendo a ação ("advance" ou "backtrack")
                                                   e a próxima coordenada (ou None se for recuar).
        """
        if valid_neighbors:
            goal = self.maze.goal_position

            # Seleciona a função de distância com base na escolha do usuário
            if self.metric == 'euclidean':
                distances = [self._euclidean_distance(n, goal) for n in valid_neighbors]
            elif self.metric == 'manhattan':
                distances = [self._minkowski_distance(n, goal, p=1) for n in valid_neighbors]
            elif self.metric == 'chebyshev':
                distances = [self._chebyshev_distance(n, goal) for n in valid_neighbors]
            else:
                distances = [self._euclidean_distance(n, goal) for n in valid_neighbors]

            min_dist = min(distances)

            # Conta quantos vizinhos empatam na distância mínima (tolerância para ponto flutuante)
            empates = sum(1 for d in distances if abs(d - min_dist) < 1e-9)
            self.tie_counts.append(empates)

            # Encontra o primeiro vizinho que atinge a distância mínima (respeitando NEIGHBOR_OFFSETS)
            next_position = valid_neighbors[distances.index(min_dist)]
            return ("advance", next_position)

        return ("backtrack", None)

    # ------------------------------------------------------------------
    # Ação
    # ------------------------------------------------------------------
    def act(self, decision: Tuple[str, Optional[Tuple[int, int]]]) -> bool:
        """
        Executa a decisão tomada, atualizando a matriz, a posição e a pilha.

        Args:
            decision (Tuple[str, Optional[Tuple[int, int]]]): A tupla gerada pelo método decide().

        Returns:
            bool: True se o agente conseguiu se mover, False se a pilha estiver vazia 
                  e não for possível recuar (sem solução).
        """
        action, target_position = decision

        if action == "advance" and target_position is not None:
            self.maze.mark_visited(*self.position)
            self.stack.append(self.position)
            self.position = target_position
            self.path.append(target_position)
            
        else:  # action == "backtrack"
            if not self.stack:
                return False
            
            self.maze.mark_visited(*self.position)
            self.position = self.stack.pop()
            self.path.append(self.position)

        self.loop_count += 1
        return True

    def final_path(self) -> List[Tuple[int, int]]:
        """
            Reconstrói o caminho efetivamente válido entre a origem e a posição atual.
        
            Returns:
                List[Tuple[int, int]]: Sequência de coordenadas da origem até a
                        posição atual, sem os trechos abandonados.
        """
        return list(self.stack) + [self.position]


    def visited_cells(self) -> int:
        """
            Conta quantas células distintas o agente percorreu durante a navegação.
        
            Returns:
                int: Quantidade de posições distintas presentes na trilha.
        """
        return len(set(self.path))

    # ------------------------------------------------------------------
    # Ciclo principal
    # ------------------------------------------------------------------
    def run(self, max_iterations: int = 1_000_000) -> bool:
        """
        Executa o ciclo Percepção -> Decisão -> Ação repetidamente até atingir o objetivo.

        Args:
            max_iterations (int, optional): Limite de segurança para evitar loops infinitos. 
                                            Padrão é 1.000.000.

        Returns:
            bool: True se encontrou o objetivo, False se esgotou as possibilidades.

        Raises:
            RuntimeError: Se o número máximo de iterações for atingido sem encontrar o objetivo.
        """
        if self.maze.is_goal(*self.position):
            self.maze.mark_visited(*self.position)
            return True

        for _ in range(max_iterations):
            valid_neighbors = self.perceive()
            decision = self.decide(valid_neighbors)
            moved = self.act(decision)

            if not moved:
                return False

            if self.maze.is_goal(*self.position):
                self.maze.mark_visited(*self.position)
                return True

        raise RuntimeError(
            "Número máximo de iterações atingido sem encontrar o objetivo."
        )