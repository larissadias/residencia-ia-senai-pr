from typing import Tuple, List, Optional

def read_matrix(file_path: str) -> Tuple[List[List[int]], Optional[Tuple[int, int]]]:
    """
    Lê um arquivo de texto contendo uma matriz e verifica se ela é quadrada (NxN).
    Detecta automaticamente se o arquivo inclui coordenadas de objetivo na última linha.

    Args:
        file_path (str): Caminho para o arquivo de entrada.

    Returns:
        Tuple[List[List[int]], Optional[Tuple[int, int]]]: Uma tupla onde o primeiro elemento 
        é a matriz (lista de listas de inteiros) e o segundo são as coordenadas do objetivo 
        (tupla) ou None se não houver.

    Raises:
        ValueError: Se a matriz estiver vazia ou não for estritamente quadrada.
    """
    
    goal_position = None
    matrix_lines = []

    with open(file_path, "r") as file:
        lines = [line.strip() for line in file if line.strip() != ""]
        
    if not lines:
        return [], None
        
    # Analisa a última linha
    last_line_values = [int(v) for v in lines[-1].split()]
    
    # Se a última linha tiver exatamente 2 valores, é o formato do desafio
    if len(last_line_values) == 2:
        goal_position = tuple(last_line_values)
        matrix_lines = lines[:-1]  # A matriz é tudo, exceto a última linha
    else:
        # Se tiver mais valores, é o formato da atividade 1
        goal_position = None
        matrix_lines = lines
        
    # Converte as linhas restantes na matriz
    matrix = [[int(val) for val in line.split()] for line in matrix_lines]

    # Garante a restrição da matriz NxN
    num_rows = len(matrix)
    if num_rows == 0:
        raise ValueError("Erro: A matriz está vazia.")
        
    for r_idx, row in enumerate(matrix):
        if len(row) != num_rows:
            raise ValueError(
                f"Erro de Formato: A matriz precisa ser estritamente quadrada (NxN). "
                f"A linha {r_idx} possui {len(row)} colunas, mas a matriz tem {num_rows} linhas."
            )

    return matrix, goal_position