import os
import platform
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from typing import Any, List, Dict


# 0 = Parede, 1 = Livre, 2 = Explorado, 3 = Caminho final, 4 = Início, 5 = Objetivo
COLORS = ['black', 'white', 'yellow', 'red', 'blue', 'green']
CMAP = ListedColormap(COLORS)
LABELS = ['Parede', 'Livre', 'Explorado', 'Caminho final', 'Início', 'Objetivo']
LEGEND = [Patch(facecolor=c, edgecolor='gray', label=l) for c, l in zip(COLORS, LABELS)]


def build_visual_matrix(base_matrix: List[List[int]], result: Dict) -> np.ndarray:
    """Monta a matriz de cores a partir da matriz original e de um resultado de busca."""
    original = np.array(base_matrix)
    visual = np.where(original == 0, 0, 1)

    for row, col in result['explored']:
        visual[row, col] = 2

    for row, col in result['path']:
        visual[row, col] = 3

    visual[result['goal'][0], result['goal'][1]] = 5
    visual[0, 0] = 4

    return visual


def _draw_panel(ax: Any, visual: np.ndarray, title: str, subtitle: str) -> None:
    """Desenha um mapa de labirinto em um dos eixos da figura."""
    ax.imshow(visual, cmap=CMAP, vmin=0, vmax=5, interpolation='nearest')

    if visual.shape[0] <= 50:
        ax.set_xticks(np.arange(-0.5, visual.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, visual.shape[0], 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

    ax.tick_params(which='both', bottom=False, left=False,
                   labelbottom=False, labelleft=False)
    ax.set_title(f"{title}\n{subtitle}", fontsize=13, pad=12)


def generate_comparison_map(base_matrix: List[List[int]], results: List[Dict], output_path: str, suptitle: str = 'Comparação entre técnicas de busca') -> str:
    """Gera a figura comparativa, com um mapa por técnica, lado a lado."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    total = len(results)
    fig, axes = plt.subplots(1, total, figsize=(7.5 * total, 8), dpi=180)
    if total == 1:
        axes = [axes]

    for ax, result in zip(axes, results):
        visual = build_visual_matrix(base_matrix, result)
        if result['success']:
            subtitle = (f"{result['steps']} passos | "
                        f"{result['expanded']} posições exploradas | "
                        f"{result['time']:.4f}s")
        else:
            subtitle = 'sem solução'
        title = f"{result['technique']} ({result['metric']})"
        _draw_panel(ax, visual, title, subtitle)

    fig.suptitle(suptitle, fontsize=17, y=0.99)
    fig.legend(handles=LEGEND, loc='lower center', ncol=6,
               frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    return output_path


def generate_metrics_chart(results: List[Dict], output_path: str) -> str:
    """Gera um gráfico de barras comparando esforço de busca e qualidade da solução."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    valid = [r for r in results if r['success']]
    labels = [f"{r['technique']}\n({r['metric']})" for r in valid]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=180)

    ax1.bar(labels, [r['expanded'] for r in valid], color='#4c72b0')
    ax1.set_title('Esforço de busca (posições exploradas)', fontsize=13)
    ax1.set_ylabel('Posições')

    ax2.bar(labels, [r['steps'] for r in valid], color='#dd8452')
    ax2.set_title('Qualidade da solução (passos do caminho)', fontsize=13)
    ax2.set_ylabel('Passos')

    for ax in (ax1, ax2):
        ax.tick_params(axis='x', labelsize=9)
        for bar in ax.patches:
            ax.annotate(f"{int(bar.get_height())}",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha='center', va='bottom', fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    return output_path


def open_images(paths: List[str]) -> None:
    """Abre as imagens geradas no visualizador padrão do sistema operacional."""
    for path in paths:
        try:
            if platform.system() == 'Windows':
                os.startfile(path)
            elif platform.system() == 'Darwin':  
                subprocess.call(['open', path])
            else: 
                subprocess.call(['xdg-open', path])
        except Exception as e:
            print(f"Não foi possível abrir '{path}' automaticamente: {e}")
