import numpy as np
import pandas as pd
from split_data import load_split, FILE_PATH

MISSING_THRESHOLD = -900

def load_train_test(file_path=FILE_PATH, apply_threshold=True):
    """Loads the dataset, applies a pre-defined train/test split based on index IDs, 
    and separates the features from the target variable 'Y'."""
    df = pd.read_csv(file_path, engine="c").set_index("ID")
    train_ids, test_ids = load_split()

    y_train = df.loc[train_ids, "Y"].astype("int8").values
    y_test = df.loc[test_ids, "Y"].astype("int8").values

    features = df.columns.drop("Y")
    feature_names = features.tolist()

    X_train = df.loc[train_ids, features].astype("float32").values
    X_test = df.loc[test_ids, features].astype("float32").values
    del df

    if apply_threshold:
        X_train[X_train <= MISSING_THRESHOLD] = np.nan
        X_test[X_test <= MISSING_THRESHOLD] = np.nan

    return X_train, X_test, y_train, y_test, feature_names


def missing_values(X_train, feature_names):
    """Calculates summary statistics for missing values (NaNs) in the dataset."""
    missing_by_column = np.isnan(X_train).mean(axis=0)
    worst = np.argsort(-missing_by_column)[:5]

    return {
        "total_percent": 100.0 * np.isnan(X_train).mean(),
        "columns_with_missing": int((missing_by_column > 0).sum()),
        "columns_above_50_percent": int((missing_by_column > 0.5).sum()),
        "columns_above_90_percent": int((missing_by_column > 0.9).sum()),
        "worst_columns": [(feature_names[i], round(100 * missing_by_column[i], 1))
                                  for i in worst],
    }

def main():
    print("Carregando o dataset segundo a divisão definida...")
    X_train, X_test, y_train, y_test, feature_names = load_train_test()

    print(f"Treino: {X_train.shape[0]} registros x {X_train.shape[1]} variáveis")
    print(f"Teste:  {X_test.shape[0]} registros x {X_test.shape[1]} variáveis")
    print(f"Classe 1 no treino: {100 * y_train.mean():.2f}%")
    print(f"Classe 1 no teste:  {100 * y_test.mean():.2f}%")

    info = missing_values(X_train, feature_names)

    print(f"\nValores ausentes no treino:")
    print(f"{'-' * 48}")
    print(f"Identificou-se que o dataset usa números menores ou iguais a "
      f"{MISSING_THRESHOLD}) para representar dados ausentes, em vez de "
      f"NaN. Então, esses valores foram convertidos para NaN, mas "
      f"ainda não foram preenchidos nessa etapa.\n")
    print(f"{info['total_percent']:.2f}% de todas as células do treino estão ausentes")
    print(f"{info['columns_with_missing']} de {len(feature_names)} colunas tem ao menos um valor ausente")
    print(f"{info['columns_above_50_percent']} colunas têm mais de 50% de seus valores ausentes")
    print(f"{info['columns_above_90_percent']} colunas têm mais de 90% de seus valores ausentes")
    print(f"\n  Colunas mais afetadas (nome / % de ausência):")
    for col, pct in info['worst_columns']:
        print(f"{col}: {pct}%")


if __name__ == "__main__":
    main()



