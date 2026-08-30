import numpy as np
import time
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier
from load_data import load_train_test
from metrics import compute_metrics, measure_latency, print_confusion
from plots import plot_error_analysis, plot_model_comparison


SEEDS = [26, 46, 90]
KNN_FEATURES = 30
CSV_PATH = "baseline_resultados.csv"


def preprocess(X_train, X_test, scale=True):
    """Preprocesses the training and test features by imputing missing values with 
    the column median and optionally applying standard scaling."""
    imputer = SimpleImputer(strategy="median")
    X_train_prep = imputer.fit_transform(X_train).astype(np.float32)
    X_test_prep = imputer.transform(X_test).astype(np.float32)

    if not scale:
        return X_train_prep, X_test_prep

    scaler = StandardScaler()
    X_train_prep = scaler.fit_transform(X_train_prep).astype(np.float32)
    X_test_prep = scaler.transform(X_test_prep).astype(np.float32)

    return X_train_prep, X_test_prep

def build_models():
    """Constructs a list of machine learning model configurations to be evaluated in the pipeline."""
    models = [
        ("Baseline classe majoritária", DummyClassifier, dict(strategy="most_frequent"),
         False, "impute"),
        ("Regressão Logística", LogisticRegression, dict(max_iter=300),
         False, "scale"),
        ("Árvore de Decisão", DecisionTreeClassifier, dict(max_depth=8),
         True, "impute"),
        ("Random Forest", RandomForestClassifier,
         dict(n_estimators=60, max_depth=12, n_jobs=-1), True, "impute"),
        ("KNN", KNeighborsClassifier, dict(n_neighbors=15, n_jobs=-1),
         False, "knn"),
        ("XGBoost", XGBClassifier,
         dict(n_estimators=100, max_depth=6, learning_rate=0.1,
              tree_method="hist", n_jobs=-1, eval_metric="logloss"),
         True, "raw"),
    ]

    return models

def train_and_evaluate(model_class, params, seed, X_train, X_test, y_train, y_test):
    """Instantiates, trains, and evaluates a machine learning model, recording 
    its performance metrics, training time, and inference latency."""
    if seed is not None:
        params = dict(params, random_state=seed)

    model = model_class(**params)
    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start_time

    result = compute_metrics(model, X_test, y_test)
    result["train_time"] = train_time
    result["latency_ms"] = measure_latency(model, X_test)

    return result


def summarize(runs, name, stochastic):
    """Aggregates performance metrics across multiple evaluation runs for a model, 
    calculating summary statistics (mean, standard deviation, min, max) and 
    identifying the best performing run based on AUC."""
    aucs = [r["auc"] for r in runs]
    balanced = [r["balanced_accuracy"] for r in runs]

    best = max(runs, key=lambda r: r["auc"])

    return {
        "name": name,
        "stochastic": stochastic,
        "n_runs": len(runs),
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
        "auc_min": min(aucs),
        "auc_max": max(aucs),
        "accuracy_mean": float(np.mean([r["accuracy"] for r in runs])),
        "balanced_mean": float(np.mean(balanced)),
        "f1_class1_mean": float(np.mean([r["f1_class1"] for r in runs])),
        "precision_class1_mean": float(np.mean([r["precision_class1"] for r in runs])),
        "recall_class1_mean": float(np.mean([r["recall_class1"] for r in runs])),
        "recall_class0_mean": float(np.mean([r["recall_class0"] for r in runs])),
        "train_time_mean": float(np.mean([r["train_time"] for r in runs])),
        "latency_mean": float(np.mean([r["latency_ms"] for r in runs])),
        "best_run": best,
    }

def print_quality_table(results):
    """Prints a formatted tabular report of quality metrics for evaluated machine 
    learning models to the standard output."""
    header = (f"{'Modelo':<30}{'AUC':>16}{'Acurácia':>10}{'Acur.Bal':>10}"
              f"{'F1(1)':>8}{'Precisão(1)':>13}{'Recall(1)':>11}{'Recall(0)':>11}")

    print(header)
    print("-" * len(header))

    for r in results:
        if r["stochastic"]:
            auc = f"{r['auc_mean']:.4f}±{r['auc_std']:.4f}"
        else:
            auc = f"{r['auc_mean']:.4f}"
        print(f"{r['name']:<30}{auc:>16}{r['accuracy_mean']:>10.4f}"
              f"{r['balanced_mean']:>10.4f}{r['f1_class1_mean']:>8.4f}"
              f"{r['precision_class1_mean']:>13.4f}{r['recall_class1_mean']:>11.4f}"
              f"{r['recall_class0_mean']:>11.4f}")


def print_cost_table(results):
    """Prints a formatted tabular report detailing the computational costs, including 
    average training time and inference latency, for the evaluated machine learning models."""
    header = f"{'Modelo':<30}{'Treino (seg)':>14}{'ms/amostra':>13}{'Execuções':>20}"
    print(header)
    print("-" * len(header))

    for r in results:
        execucoes = f"{r['n_runs']} sementes" if r["stochastic"] else "1 (determinístico)"
        print(f"{r['name']:<30}{r['train_time_mean']:>14.1f}"
              f"{r['latency_mean']:>13.3f}{execucoes:>20}")


def save_csv(results, path=CSV_PATH):
    """Exports the summarized model evaluation results to a CSV file."""
    columns = ["modelo", "execucoes", "auc_media", "auc_desvio",
               "auc_min", "auc_max", "acuracia", "acuracia_balanceada",
               "f1_classe1", "precisao_classe1", "recall_classe1",
               "recall_classe0", "tempo_treino_s", "latencia_ms"]
    

    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(columns) + "\n")
        for r in results:
            f.write(f"{r['name']},{r['n_runs']},{r['auc_mean']:.4f},{r['auc_std']:.4f},"
                    f"{r['auc_min']:.4f},{r['auc_max']:.4f},{r['accuracy_mean']:.4f},"
                    f"{r['balanced_mean']:.4f},{r['f1_class1_mean']:.4f},"
                    f"{r['precision_class1_mean']:.4f},{r['recall_class1_mean']:.4f},"
                    f"{r['recall_class0_mean']:.4f},{r['train_time_mean']:.2f},"
                    f"{r['latency_mean']:.4f}\n")

    return path

def main():
    print("Carregando os conjuntos conforme a divisão registrada...")
    X_train, X_test, y_train, y_test, feature_names = load_train_test()
    print(f"Treino: {X_train.shape} | Teste: {X_test.shape}")
    print(f"Classe 1: {100 * y_train.mean():.2f}% no treino, "
          f"{100 * y_test.mean():.2f}% no teste\n")
    
    models = build_models()
    results_by_name = {}

    for data_key in ("raw", "impute", "scale", "knn"):
        grupo = [m for m in models if m[4] == data_key]
        if not grupo:
            continue

        if data_key == "raw":
            X_train_ready, X_test_ready = np.asarray(X_train), np.asarray(X_test)
        elif data_key == "impute":
            print("Preparando dados imputados (mediana do treino)...")
            X_train_ready, X_test_ready = preprocess(X_train, X_test, scale=False)
        elif data_key == "scale":
            print("Preparando dados imputados e padronizados...")
            X_train_ready, X_test_ready = preprocess(X_train, X_test, scale=True)
        else:
            print(f"Preparando subconjunto de {KNN_FEATURES} variáveis para o KNN...")
            X_train_scaled, X_test_scaled = preprocess(X_train.copy(), X_test.copy(), scale=True)
            selector = SelectKBest(score_func=f_classif, k=KNN_FEATURES)
            X_train_ready = selector.fit_transform(X_train_scaled, y_train)
            X_test_ready = selector.transform(X_test_scaled)

        for name, model_class, params, stochastic, i in grupo:
            seeds = SEEDS if stochastic else [None]
            print(f"Treinando {name}...")
            runs = []
            for seed in seeds:
                run = train_and_evaluate(model_class, params, seed, X_train_ready, X_test_ready, y_train, y_test)
                runs.append(run)
                marca = f"semente {seed}" if seed is not None else "determinístico"
                print(f"  {marca}: AUC {run['auc']:.4f} | {run['train_time']:.1f}s")
            results_by_name[name] = summarize(runs, name, stochastic)

    results = [results_by_name[m[0]] for m in models]

    print("\n" + "=" * 95)
    print("FASE 1 - BASELINE: EXPERIMENTO QUALIDADE")
    print("=" * 95)
    print_quality_table(results)

    print("\n" + "=" * 95)
    print("FASE 1 - BASELINE: EXPERIMENTO CUSTO COMPUTACIONAL")
    print("=" * 95)
    print_cost_table(results)

    candidates = results[1:]
    best = max(candidates, key=lambda r: r["auc_mean"])
    print_confusion(y_test, best["best_run"]["predictions"],
                    f"{best['name']} (melhor execução)")

    path = save_csv(results)
    print(f"\nTabela salva em {path}")
    print(f"  {plot_model_comparison(results)}")
    print(f"  {plot_error_analysis(y_test, best['best_run']['probabilities'])}")


if __name__ == "__main__":
    main()
