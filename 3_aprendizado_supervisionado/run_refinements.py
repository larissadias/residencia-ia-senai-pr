import time
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,f1_score, recall_score)
from xgboost import XGBClassifier
from load_data import load_train_test
from metrics import best_threshold, compute_metrics, measure_latency, print_confusion
from plots import plot_error_analysis, plot_threshold_effect

SEED = 26
CSV_PATH = "refinements_results.csv"

BASELINE_PARAMS = dict(n_estimators=100, max_depth=6, learning_rate=0.1,
                        tree_method="hist", n_jobs=-1, eval_metric="logloss", random_state=SEED)


def evaluate(X_train, X_test, y_train, y_test, params=None, label=""):
    """Trains the models and returns its metrics."""
    params = params or BASELINE_PARAMS

    start_time = time.perf_counter()
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start_time

    result = compute_metrics(model, X_test, y_test)
    result["label"] = label
    result["n_features"] = X_train.shape[1]
    result["train_time"] = train_time
    result["latency_ms"] = measure_latency(model, X_test)

    return result


def print_comparison(baseline, results):
    """Prints each experiment against the baseline."""
    header = (f"{'Experimento':<34}{'Variáveis':>10}{'AUC':>9}{'vs base':>10}"
              f"{'Acur.Bal':>10}{'Recall(0)':>11}{'Treino':>9}")
    print(header)
    print("-" * len(header))

    print(f"{baseline['label']:<34}{baseline['n_features']:>10}"
          f"{baseline['auc']:>9.4f}{'—':>10}{baseline['balanced_accuracy']:>10.4f}"
          f"{baseline['recall_class0']:>11.4f}{baseline['train_time']:>8.1f}s")

    for r in results:
        delta = r["auc"] - baseline["auc"]
        print(f"{r['label']:<34}{r['n_features']:>10}{r['auc']:>9.4f}"
              f"{delta:>+10.4f}{r['balanced_accuracy']:>10.4f}"
              f"{r['recall_class0']:>11.4f}{r['train_time']:>8.1f}s")


def save_csv(baseline, results, path=CSV_PATH):
    """Saves the comparison table."""
    columns = ["experimento", "variaveis", "auc", "diferenca_vs_base",
               "acuracia", "acuracia_balanceada", "f1_classe1", "recall_classe0",
               "tempo_treino_s", "latencia_ms"]

    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(columns) + "\n")
        for r in [baseline] + results:
            delta = r["auc"] - baseline["auc"]
            f.write(f"{r['label']},{r['n_features']},{r['auc']:.4f},{delta:+.4f},"
                    f"{r['accuracy']:.4f},{r['balanced_accuracy']:.4f},{r['f1_class1']:.4f},"
                    f"{r['recall_class0']:.4f},{r['train_time']:.2f},"
                    f"{r['latency_ms']:.4f}\n")
            
    return path


def main():
    print("Carregando os conjuntos conforme a divisão registrada...")
    X_train, X_test, y_train, y_test, feature_names = load_train_test()
    print(f"Treino: {X_train.shape} | Teste: {X_test.shape}\n")

    print("=" * 92)
    print("MELHOR MODELO BASELINE")
    print("=" * 92)
    baseline = evaluate(X_train, X_test, y_train, y_test, label="Baseline: XGBoost - 672 vars")
    print(f"AUC {baseline['auc']:.4f} | acurácia balanceada {baseline['balanced_accuracy']:.4f} "
              f"| recall da classe 0: {baseline['recall_class0']:.4f}\n")
    
    results = []

    print("\nEXPERIMENTO 1 - Ajuste de Hiperparâmetros")
    print("=" * 92)

    configs = [
        ("Mais árvores (300)", dict(BASELINE_PARAMS, n_estimators=300)),
        ("Árvores mais profundas (10)", dict(BASELINE_PARAMS, max_depth=10)),
        ("Amostragem 0.8", dict(BASELINE_PARAMS, subsample=0.8, colsample_bytree=0.8)),
        ("Peso por classe", dict(BASELINE_PARAMS, scale_pos_weight=0.2576)),
    ]

    for label, params in configs:
        r = evaluate(X_train, X_test, y_train, y_test, params=params, label=label)
        results.append(r)
        print(f"  {label:<32} AUC {r['auc']:.4f} ({r['auc'] - baseline['auc']:+.4f}) "
              f"| acur.bal {r['balanced_accuracy']:.4f}")

    print("\n" + "=" * 92)
    print("COMPARAÇÃO CONTRA BASELINE")
    print("=" * 92)
    print_comparison(baseline, results)

    best = max(results, key=lambda r: r["auc"])
    print(f"\nMelhor experimento por AUC: {best['label']} "
          f"({best['auc']:.4f}, {best['auc'] - baseline['auc']:+.4f})")



    print("\n" + "=" * 92)
    print("EXPERIMENTO 2 - Ajuste do Limiar de Decisão")
    print("=" * 92)

    reference = best if best["auc"] > baseline["auc"] else baseline
    threshold = best_threshold(y_test, reference["probabilities"])
    adjusted = (reference["probabilities"] >= threshold).astype(int)

    print(f"Aplicado sobre: {reference['label']}")
    print(f"Limiar ajustado: {threshold:.4f} (o padrão é 0,5)\n")
    print(f"{'':<24}{'limiar 0,5':>13}{'limiar ajustado':>18}")
    print(f"{'AUC':<24}{reference['auc']:>13.4f}{reference['auc']:>18.4f}")
    print(f"{'Acurácia':<24}{reference['accuracy']:>13.4f}"
          f"{accuracy_score(y_test, adjusted):>18.4f}")
    print(f"{'Acurácia balanceada':<24}{reference['balanced_accuracy']:>13.4f}"
          f"{balanced_accuracy_score(y_test, adjusted):>18.4f}")
    print(f"{'Recall da classe 0':<24}{reference['recall_class0']:>13.4f}"
          f"{recall_score(y_test, adjusted, pos_label=0, zero_division=0):>18.4f}")
    print(f"{'F1 da classe 1':<24}{reference['f1_class1']:>13.4f}"
          f"{f1_score(y_test, adjusted, zero_division=0):>18.4f}")

    print_confusion(y_test, reference["predictions"], "limiar padrão de 0,5")
    print_confusion(y_test, adjusted, f"limiar ajustado de {threshold:.2f}")

    path = save_csv(baseline, results)
    print(f"\nTabela salva em {path}")
    
    print("\nGerando figuras...")
    print(f"  {plot_threshold_effect(y_test, reference['probabilities'], threshold)}")
    print(f"  {plot_error_analysis(y_test, reference['probabilities'], threshold)}")


if __name__ == "__main__":
        main()