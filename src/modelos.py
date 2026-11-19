
from copy import deepcopy

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import RANDOM_STATE

NOMES_MODELOS = ("DT", "RF", "SVM", "XGBoost", "KNN")

# Configurações dos experimentos iniciais, antes da busca de hiperparâmetros.
PARAMETROS_BASE = {
    "DT": {"max_depth": 15, "min_samples_leaf": 1, "min_samples_split": 2},
    "RF": {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1},
    "SVM": {"C": 1.0},
    "XGBoost": {
        "n_estimators": 150,
        "max_depth": 4,
        "learning_rate": 0.05,
        "min_child_weight": 1,
    },
    "KNN": {"n_neighbors": 5, "weights": "uniform"},
}

# Configurações congeladas após o Experimento 5.
# Não devem ser alteradas com base nos resultados do conjunto de teste.
CONFIGURACOES_FINAIS = (
    {
        "tarefa": "Multiclasse",
        "modelo": "XGBoost",
        "representacao": "Word",
        "balanceamento": "Sem",
        "parametros": {
            "n_estimators": 150, "max_depth": 4,
            "learning_rate": 0.05, "min_child_weight": 1,
        },
    },
    {
        "tarefa": "Multiclasse",
        "modelo": "DT",
        "representacao": "Word",
        "balanceamento": "Pesos",
        "parametros": {
            "max_depth": 15, "min_samples_leaf": 1,
            "min_samples_split": 2,
        },
    },
    {
        "tarefa": "Binario",
        "modelo": "XGBoost",
        "representacao": "Combined",
        "balanceamento": "Sem",
        "parametros": {
            "n_estimators": 100, "max_depth": 2,
            "learning_rate": 0.1, "min_child_weight": 3,
        },
    },
    {
        "tarefa": "Binario",
        "modelo": "SVM",
        "representacao": "Combined",
        "balanceamento": "Pesos",
        "parametros": {"C": 1.0},
    },
)

# Cada chamada cria um objeto novo e ainda não ajustado.
def criar_word_tfidf():
    return TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=1,
        sublinear_tf=True, norm="l2",
    )


def criar_char_tfidf():
    return TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=1,
        sublinear_tf=True, norm="l2",
    )


def criar_representacao(nome="Word"):
    """Cria Word, Character ou Combined (aceita aliases dos experimentos)."""
    aliases = {
        "Word": "Word", "R1_Word": "Word",
        "Character": "Character", "R2_Character": "Character",
        "Combined": "Combined", "R3_Combined": "Combined",
    }
    if nome not in aliases:
        raise ValueError(f"Representação inválida: {nome!r}")
    representacao = aliases[nome]
    if representacao == "Word":
        return criar_word_tfidf()
    if representacao == "Character":
        return criar_char_tfidf()
    return Pipeline([
        ("union", FeatureUnion([
            ("word", criar_word_tfidf()),
            ("char", criar_char_tfidf()),
        ])),
        ("normalizer", Normalizer(norm="l2")),
    ])


def criar_classificador(nome, n_classes, parametros=None, balanceamento="Baseline"):
    """Cria um classificador sem ajustar.

    balanceamento='Baseline': DT/RF/SVM usam class_weight='balanced',
    como nos Experimentos 1–3; XGBoost/KNN não recebem pesos.

    balanceamento='Sem' ou 'Pesos': não define class_weight. Para 'Pesos',
    o script chamador deve calcular os pesos no treino de cada fold e
    fornecê-los via sample_weight ao método fit().

    balanceamento='Oversampling': não define pesos. O script chamador
    aplica a reamostragem somente ao treino do fold antes do fit().
    """
    if nome not in NOMES_MODELOS and nome != "Dummy":
        raise ValueError(f"Modelo inválido: {nome!r}")
    if n_classes not in (2, 3):
        raise ValueError("n_classes deve ser 2 ou 3")
    if balanceamento not in ("Baseline", "Sem", "Pesos", "Oversampling"):
        raise ValueError(f"Balanceamento inválido: {balanceamento!r}")
    if nome == "Dummy":
        if parametros:
            raise ValueError("Dummy não aceita parâmetros adicionais nesta fábrica")
        return DummyClassifier(strategy="most_frequent")

    params = deepcopy(PARAMETROS_BASE[nome])
    if parametros is not None:
        params.update(parametros)

    if nome in ("DT", "RF", "SVM") and balanceamento == "Baseline":
        params["class_weight"] = "balanced"

    if nome == "DT":
        return DecisionTreeClassifier(**params, random_state=RANDOM_STATE)
    if nome == "RF":
        return RandomForestClassifier(
            **params, random_state=RANDOM_STATE, n_jobs=1
        )
    if nome == "SVM":
        return LinearSVC(
            **params, random_state=RANDOM_STATE, max_iter=5000
        )
    if nome == "KNN":
        return KNeighborsClassifier(
            **params, metric="cosine", algorithm="brute"
        )

    return XGBClassifier(
        **params,
        objective="multi:softprob" if n_classes == 3 else "binary:logistic",
        eval_metric="mlogloss" if n_classes == 3 else "logloss",
        tree_method="hist", random_state=RANDOM_STATE, n_jobs=1,
    )


def criar_pipeline(nome, n_classes, representacao="Word",
                   parametros=None, balanceamento="Baseline"):
    """TF-IDF e classificador no mesmo pipeline (sem vazamento entre folds)."""
    return Pipeline([
        ("features", criar_representacao(representacao)),
        ("classifier", criar_classificador(
            nome, n_classes, parametros=parametros,
            balanceamento=balanceamento,
        )),
    ])
