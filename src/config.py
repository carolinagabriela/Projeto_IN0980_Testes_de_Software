
from pathlib import Path

# DIRETÓRIO PRINCIPAL

BASE_DIR = Path(__file__).resolve().parents[1]

# DIRETÓRIOS DE DADOS

DATA_DIR = BASE_DIR / "data"

RAW_DIR = DATA_DIR / "raw"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
PROCESSED_DIR = DATA_DIR / "processed"
REVIEW_DIR = DATA_DIR / "review"

RESULTS_DIR = BASE_DIR / "results"

# ARQUIVOS DE REFERÊNCIA

DATASET_ORIGINAL = (
    RAW_DIR / "valid_dataset.xlsx"
)

DATASET_GITHUB = (
    SNAPSHOT_DIR / "github_issues_original.xlsx"
)

DATASET_LIMPO = (
    PROCESSED_DIR / "dataset_limpo.xlsx"
)

DATASET_TREINO = (
    PROCESSED_DIR / "dataset_treino.xlsx"
)

DATASET_TESTE = (
    PROCESSED_DIR / "dataset_teste.xlsx"
)

# CONFIGURAÇÕES EXPERIMENTAIS

RANDOM_STATE = 42

N_FOLDS = 5

TEST_SIZE = 0.20

CLASSES = [
    "Overlap",
    "Truncation",
    "Ellipsis"
]

LABELS_MULTICLASSE = {
    "Overlap": 0,
    "Truncation": 1,
    "Ellipsis": 2
}

LABELS_BINARIO = {
    "Overlap": 0,
    "Truncation": 1,
    "Ellipsis": 1
}
