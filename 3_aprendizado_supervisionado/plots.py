
import numpy as np
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

AZUL = "#4c72b0"
LARANJA = "orange"
VERMELHO = "red"
VERDE = "green"


def plot_model_comparison(results, path="figura_comparacao_modelos.png"):
    """Compares the models on accuracy and on AUC, side by side."""
    names = [r["name"] for r in results]
    accuracies = [r["accuracy_mean"] for r in results]
    aucs = [r["auc_mean"] for r in results]
    balanced = [r["balanced_mean"] for r in results]
    baseline_accuracy = results[0]["accuracy_mean"]

    positions = np.arange(len(names))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=150)

    ax1.bar(positions, accuracies, color=VERMELHO)
    ax1.axhline(baseline_accuracy, color="black", linestyle="--", linewidth=1.5,
                label=f"Responder sempre a classe majoritária ({baseline_accuracy:.3f})")
    ax1.set_ylim(min(accuracies) - 0.02, max(accuracies) + 0.02)
    ax1.set_title("Acurácia: todos os modelos parecem equivalentes", fontsize=12)
    ax1.set_ylabel("Acurácia")
    ax1.legend(fontsize=9, loc="lower right")

    largura = 0.38
    ax2.bar(positions - largura / 2, aucs, largura, color=AZUL, label="ROC AUC")
    ax2.bar(positions + largura / 2, balanced, largura, color=LARANJA,
            label="Acurácia balanceada")
    ax2.axhline(0.5, color="black", linestyle="--", linewidth=1.5, label="Acaso (0,5)")
    ax2.set_ylim(0.4, max(aucs) + 0.08)
    ax2.set_title("AUC e acurácia balanceada: a diferença aparece", fontsize=12)
    ax2.set_ylabel("Valor da métrica")
    ax2.legend(fontsize=9, loc="upper left")

    for ax, valores in ((ax1, accuracies), (ax2, aucs)):
        ax.set_xticks(positions)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(valores):
            ax.annotate(f"{v:.3f}", (i, v), ha="center", va="bottom", fontsize=8)

    fig.suptitle("Por que a acurácia não serve como métrica principal "
                 "em base desbalanceada", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return path


def plot_error_analysis(y_test, probabilities, threshold=None, path="figura_analise_erros.png"):
    """Shows how the predicted probabilities of the two classes overlap."""
    prob_class0 = probabilities[y_test == 0]
    prob_class1 = probabilities[y_test == 1]

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)

    bins = np.linspace(0, 1, 51)
    ax.hist(prob_class1, bins=bins, alpha=0.6, color=AZUL, density=True,
            label=f"Classe 1 ({len(prob_class1)} casos)")
    ax.hist(prob_class0, bins=bins, alpha=0.6, color=VERMELHO, density=True,
            label=f"Classe 0 ({len(prob_class0)} casos)")

    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.5, label="Limiar padrão (0,50)")

    if threshold is not None:
        ax.axvline(threshold, color=VERDE, linestyle="-", linewidth=2, label=f"Limiar ajustado ({threshold:.2f})")

    ax.set_xlabel("Probabilidade atribuída pelo modelo à classe 1")
    ax.set_ylabel("Densidade")
    ax.set_title("Análise de erros: as duas classes ocupam a mesma faixa "
                 "de probabilidade", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return path


def _draw_confusion(ax, y_test, predictions, title):
    """Draws one confusion matrix, with each row normalized by its total."""
    matrix = confusion_matrix(y_test, predictions)
    normalized = matrix / matrix.sum(axis=1, keepdims=True)

    ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)

    for i in range(2):
        for j in range(2):
            cor = "white" if normalized[i, j] > 0.5 else "black"
            ax.text(j, i, f"{matrix[i, j]}\n({100 * normalized[i, j]:.1f}%)",
                    ha="center", va="center", color=cor, fontsize=12)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["previsto 0", "previsto 1"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["real 0", "real 1"])
    ax.set_title(title, fontsize=12)


def plot_threshold_effect(y_test, probabilities, threshold, path="figura_limiar.png"):
    """Shows the ROC curve and the confusion matrices before and after the threshold."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=150)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, probabilities)
    auc = roc_auc_score(y_test, probabilities)

    axes[0].plot(false_positive_rate, true_positive_rate, color=AZUL, linewidth=2,
                 label=f"Modelo (AUC = {auc:.4f})")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, label="Acaso (AUC = 0,5)")

    # Marca onde ficam os dois limiares sobre a curva
    for valor, cor, rotulo in ((0.5, VERMELHO, "limiar 0,50"),
                               (threshold, VERDE, f"limiar {threshold:.2f}")):
        indice = int(np.argmin(np.abs(thresholds - valor)))
        axes[0].scatter([false_positive_rate[indice]], [true_positive_rate[indice]],
                        s=90, color=cor, zorder=3, label=rotulo)

    axes[0].set_xlabel("Taxa de falsos positivos")
    axes[0].set_ylabel("Taxa de verdadeiros positivos")
    axes[0].set_title("Curva ROC", fontsize=12)
    axes[0].legend(fontsize=9, loc="lower right")
    axes[0].grid(alpha=0.3)

    _draw_confusion(axes[1], y_test, (probabilities >= 0.5).astype(int),
                    "Limiar padrão (0,50)")
    _draw_confusion(axes[2], y_test, (probabilities >= threshold).astype(int),
                    f"Limiar ajustado ({threshold:.2f})")

    fig.suptitle("O modelo ordena bem, mas é o limiar que define o que ele decide",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return path