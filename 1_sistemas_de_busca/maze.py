from typing import List, Tuple, Optional

class Maze:
    """
    Representa a grade do labirinto e as regras de movimentação do agente.
    """

    VISITED = 255
    BLOCKED = 0
    GOAL = -1

    def __init__(self, matrix: List[List[int]], goal_position: Optional[Tuple[int, int]] = None):
        """
        Inicializa o ambiente do labirinto.

        Args:
            matrix (List[List[int]]): A matriz bidimensional representando o labirinto.
            goal_position (Tuple[int, int], optional): Coordenadas (linha, coluna) do objetivo. 
                                                       Se None, o objetivo será o canto inferior direito.
        """
        self.matrix = matrix
        self.num_rows = len(matrix)
        self.num_cols = len(matrix[0])
        
        if goal_position is not None:
            # Formato do Desafio
            self.goal_position = goal_position
        else:
            # Formato da Atividade 1
            self.goal_position = self._locate_goal_marker()

        r, c = self.goal_position
        if not self.is_inside_bounds(r, c):
            raise ValueError(
                f"Objetivo em {self.goal_position} está fora dos limites da matriz "
                f"{self.num_rows}x{self.num_cols}."
            )

        self.matrix[r][c] = self.GOAL


    def _locate_goal_marker(self) -> Tuple[int, int]:
        """
        Localiza a célula marcada com valor negativo na matriz.

        Returns:
            Tuple[int, int]: Coordenadas (linha, coluna) do objetivo. Caso
                nenhuma marcação negativa exista, devolve o canto inferior
                direito como convenção padrão.
        """
        for r_idx, row in enumerate(self.matrix):
            for c_idx, value in enumerate(row):
                if value < 0:
                    return (r_idx, c_idx)
        return (self.num_rows - 1, self.num_cols - 1)


    def is_inside_bounds(self, row: int, col: int) -> bool:
        """
        Verifica se uma posição (linha, coluna) está estritamente dentro dos limites da matriz.

        Args:
            row (int): O índice da linha.
            col (int): O índice da coluna.

        Returns:
            bool: True se a coordenada for válida, False caso contrário.
        """
        return 0 <= row < self.num_rows and 0 <= col < self.num_cols


    def is_walkable(self, row: int, col: int) -> bool:
        """
        Verifica se uma posição pode ser acessada pelo agente.
        
        Uma posição é transitável se estiver dentro dos limites da matriz e 
        seu valor for livre (1) ou o objetivo (-1).
        
        Args:
            row (int): O índice da linha a ser verificada.
            col (int): O índice da coluna a ser verificada.
            
        Returns:
            bool: True se a posição for transitável, False caso contrário.
        """
        if not self.is_inside_bounds(row, col):
            return False
        return self.matrix[row][col] in (1, self.GOAL)


    def get_value(self, row: int, col: int) -> int:
        """
        Obtém o valor numérico atual de uma célula específica na matriz.

        Args:
            row (int): O índice da linha.
            col (int): O índice da coluna.

        Returns:
            int: O valor contido na posição especificada.
        """
        return self.matrix[row][col]


    def mark_visited(self, row: int, col: int) -> None:
        """
        Marca uma posição como visitada (valor 255).
        
        Inclui a posição final para alinhar com o estado de exemplo 
        da matriz apresentado nos slides da disciplina.

        Args:
            row (int): O índice da linha.
            col (int): O índice da coluna.
        """
        self.matrix[row][col] = self.VISITED


    def is_goal(self, row: int, col: int) -> bool:
        """
        Compara as coordenadas fornecidas com a posição de destino estabelecida.

        Args:
            row (int): O índice da linha.
            col (int): O índice da coluna.

        Returns:
            bool: True se a coordenada for o objetivo final, False caso contrário.
        """
        return (row, col) == self.goal_position