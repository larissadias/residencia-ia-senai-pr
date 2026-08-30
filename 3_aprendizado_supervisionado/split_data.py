import os
import numpy as np
import pandas as pd

TEST_SIZE = 0.2
SPLIT_SEED = 26
TRAIN_FILE = "treino_ids.txt"
TEST_FILE = "teste_ids.txt"
FILE_PATH = "data/dataset.csv"

def create_split(file_path=FILE_PATH, test_size=TEST_SIZE, seed=SPLIT_SEED):
    """Creates a stratified train/test split of dataset IDs based on the target variable 'Y', 
    ensuring proportional class representation, and saves the resulting IDs to text files."""
    df = pd.read_csv(file_path, usecols=["ID","Y"])

    generator = np.random.RandomState(seed)
    train_ids = []
    test_ids = []

    for label in sorted(df["Y"].unique()):
        ids = df.loc[df["Y"] == label, "ID"].values.copy()
        generator.shuffle(ids)

        cut = int(round(len(ids)* test_size))
        test_ids.extend(ids[:cut])
        train_ids.extend(ids[cut:])

    train_ids = np.array(sorted(train_ids))
    test_ids = np.array(sorted(test_ids))

    np.savetxt(TRAIN_FILE, train_ids, fmt="%d")
    np.savetxt(TEST_FILE, test_ids, fmt="%d")

    return train_ids, test_ids

def load_split():
    """Loads the dataset split IDs for the training and testing sets from previously 
    saved text files."""
    if not os.path.exists(TRAIN_FILE) or not os.path.exists(TEST_FILE):
        raise FileNotFoundError(
            f"Arquivos '{TRAIN_FILE}' e '{TEST_FILE}' não encontrados. "
            f"Rode create_split() antes."
        )

    train_ids = np.loadtxt(TRAIN_FILE, dtype=np.int64)
    test_ids = np.loadtxt(TEST_FILE, dtype=np.int64)

    return train_ids, test_ids

def split_info(file_path=FILE_PATH):
    """Analyzes the current train/test split and returns summary statistics, including 
    set sizes, class distributions, and data integrity checks."""
    train_ids, test_ids = load_split()
    df = pd.read_csv(file_path, usecols=["ID", "Y"]).set_index("ID")

    y_train = df.loc[train_ids, "Y"]
    y_test = df.loc[test_ids, "Y"]

    total = len(train_ids) + len(test_ids)
    overlap = len(set(train_ids) & set(test_ids))

    return {
        "train_size": len(train_ids),
        "test_size": len(test_ids),
        "test_pct": 100.0 * len(test_ids) / total,
        "train_positive_pct": 100.0 * y_train.mean(),
        "test_positive_pct": 100.0 * y_test.mean(),
        "overlap": overlap,
        "covers_all": total == len(df),
    }

def main():
    print("Dividindo o dataset em treino/teste...")
    train_ids, test_ids = create_split()
    print(f"Arquivos gravados: {TRAIN_FILE} ({len(train_ids)} IDs) e "
          f"{TEST_FILE} ({len(test_ids)} IDs)\n")

    info = split_info()
    print(f"Treino: {info['train_size']} registros | {info['train_positive_pct']:.2f}% classe 1")
    print(f"Teste:  {info['test_size']} registros | {info['test_positive_pct']:.2f}% classe 1")
    print(f"Proporção de teste: {info['test_pct']:.1f}%")
    print(f"\nVerificações:")
    print(f"  registros em comum entre treino e teste: {info['overlap']}")
    print(f"  a divisão cobre todo o dataset: {info['covers_all']}")

if __name__ == "__main__":
    main() 