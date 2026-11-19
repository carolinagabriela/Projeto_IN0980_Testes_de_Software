
from pathlib import Path
from time import perf_counter
import json
import sys

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score,
)
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

from src.config import (
    DATASET_TREINO, LABELS_BINARIO, LABELS_MULTICLASSE,
    N_FOLDS, RANDOM_STATE, RESULTS_DIR,
)
from src.modelos import (
    NOMES_MODELOS, PARAMETROS_BASE,
    criar_classificador, criar_representacao,
)

SAIDA = RESULTS_DIR / "ajuste_hiperparametros"
MAX_COMBINACOES = 10
METRICAS = (
    "f1_macro", "precision_macro", "recall_macro",
    "accuracy", "balanced_accuracy", "fit_time_s", "predict_time_s",
)
DISTRIBUICAO_ESPERADA = {"Overlap": 185, "Truncation": 113, "Ellipsis": 20}

# Configurações selecionadas antes do ajuste, com base no Experimento 4.
TAREFAS = {
    "Multiclasse": {
        "representacao": "Word",
        "balanceamento": {
            "DT": "Pesos", "RF": "Sem", "SVM": "Oversampling",
            "XGBoost": "Sem", "KNN": "Sem",
        },
        "nomes_classes": ["Overlap", "Truncation", "Ellipsis"],
    },
    "Binario": {
        "representacao": "Combined",
        "balanceamento": {
            "DT": "Pesos", "RF": "Pesos", "SVM": "Pesos",
            "XGBoost": "Sem", "KNN": "Sem",
        },
        "nomes_classes": ["Overlap", "Truncation/Ellipsis"],
    },
}

# Exatamente o mesmo espaço discreto do script histórico.
ESPACOS = {
    "DT": {
        "max_depth": [5, 10, 15, None],
        "min_samples_leaf": [1, 3, 5],
        "min_samples_split": [2, 5],
    },
    "RF": {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_leaf": [1, 3],
    },
    "SVM": {"C": [0.01, 0.1, 1.0, 10.0]},
    "XGBoost": {
        "n_estimators": [100, 150, 250],
        "max_depth": [2, 4],
        "learning_rate": [0.05, 0.1],
        "min_child_weight": [1, 3],
    },
    "KNN": {
        "n_neighbors": [3, 5, 7, 11],
        "weights": ["uniform", "distance"],
    },
}


def carregar_dados():
    if not DATASET_TREINO.is_file():
        raise FileNotFoundError(f"Dataset de treinamento não encontrado: {DATASET_TREINO}")
    df = pd.read_excel(DATASET_TREINO)
    colunas = {"url", "bug type", "title", "body"}
    if not colunas.issubset(df.columns):
        raise ValueError(f"Colunas ausentes: {sorted(colunas - set(df.columns))}")
    if len(df) != 318 or df["bug type"].value_counts().to_dict() != DISTRIBUICAO_ESPERADA:
        raise ValueError("Quantidade ou distribuição de classes diferente da histórica")
    if df["url"].isna().any() or df["url"].duplicated().any():
        raise ValueError("URLs ausentes ou duplicadas")
    texto = (
        df["title"].fillna("").astype(str) + " " +
        df["body"].fillna("").astype(str)
    ).str.strip().reset_index(drop=True)
    if texto.eq("").any():
        raise ValueError("Existem issues sem título e body")
    return df, texto


def selecionar_combinacoes(nome, seed):
    # A baseline é a configuração base do modelos.py (copiada, nunca alterada).
    baseline = dict(PARAMETROS_BASE[nome])
    todas = list(ParameterGrid(ESPACOS[nome]))
    alternativas = [p for p in todas if p != baseline]
    quantidade = min(MAX_COMBINACOES - 1, len(alternativas))
    selecionadas = []
    if quantidade:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(alternativas), size=quantidade, replace=False)
        selecionadas = [alternativas[int(i)] for i in indices]
    return [baseline] + selecionadas


def oversampling(X, y, seed):
    """Replica o algoritmo histórico; opera apenas no treino de cada fold."""
    rng = np.random.default_rng(seed)
    classes, contagens = np.unique(y, return_counts=True)
    alvo = int(contagens.max())
    indices = []
    for classe in classes:
        originais = np.flatnonzero(y == classe)
        extras = rng.choice(originais, size=alvo - len(originais), replace=True)
        indices.extend(originais.tolist())
        indices.extend(extras.tolist())
    indices = np.asarray(indices, dtype=int)
    rng.shuffle(indices)
    return X[indices], y[indices]


def verificar_referencia(resumo):
    caminho = SAIDA / "referencia_todas_configuracoes.csv"
    if not caminho.exists():
        print("\nReferência histórica opcional não encontrada; comparação ignorada.")
        return
    historico = pd.read_csv(caminho)
    chaves = ["tarefa", "modelo", "representacao", "balanceamento", "config_id"]
    comparacao = resumo.merge(historico, on=chaves, suffixes=("_novo", "_historico"),
                              how="outer", indicator=True, validate="one_to_one")
    if not comparacao["_merge"].eq("both").all():
        raise AssertionError("Conjunto de configurações difere da referência histórica")
    for campo in ("parametros",):
        if not (comparacao[f"{campo}_novo"] == comparacao[f"{campo}_historico"]).all():
            raise AssertionError(f"Parâmetros diferentes da referência: {campo}")
    for metrica in METRICAS:
        if metrica.endswith("_time_s"):
            continue
        for estatistica in ("mean", "std"):
            campo = f"{metrica}_{estatistica}"
            dif = (comparacao[f"{campo}_novo"] -
                   comparacao[f"{campo}_historico"]).abs().max()
            print(f"  {campo}: diferença máxima = {dif:.9f}")
            if not np.isfinite(dif) or dif > 1e-6:
                raise AssertionError(f"Divergência em {campo}: {dif}")
    print("Reprodução histórica aprovada (tolerância de 1e-6).")


def executar():
    df, texto = carregar_dados()
    y_multi = df["bug type"].map(LABELS_MULTICLASSE).to_numpy(dtype=int)
    y_bin = df["bug type"].map(LABELS_BINARIO).to_numpy(dtype=int)
    ys = {"Multiclasse": y_multi, "Binario": y_bin}

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(cv.split(texto, y_multi))
    SAIDA.mkdir(parents=True, exist_ok=True)

    # Reaproveitamento seguro: o TF-IDF é ajustado só no treino de cada fold.
    cache = {}
    for tarefa, configuracao in TAREFAS.items():
        y = ys[tarefa]
        nome_rep = configuracao["representacao"]
        conjuntos = []
        for numero_fold, (idx_tr, idx_val) in enumerate(folds, 1):
            representacao = criar_representacao(nome_rep)
            X_tr = representacao.fit_transform(texto.iloc[idx_tr])
            X_val = representacao.transform(texto.iloc[idx_val])
            conjuntos.append((numero_fold, X_tr, X_val, y[idx_tr], y[idx_val]))
        cache[tarefa] = conjuntos

    metricas_folds = []
    metricas_classes = []
    ordem_modelos = list(NOMES_MODELOS)
    if ordem_modelos != ["DT", "RF", "SVM", "XGBoost", "KNN"]:
        raise AssertionError("Ordem dos modelos mudou; isso alteraria a amostragem histórica")

    for tarefa, configuracao in TAREFAS.items():
        y = ys[tarefa]
        nomes_classes = configuracao["nomes_classes"]
        labels = list(range(len(nomes_classes)))
        representacao = configuracao["representacao"]
        print(f"\n========== {tarefa} | {representacao} ==========", flush=True)

        for numero_modelo, nome in enumerate(ordem_modelos):
            estrategia = configuracao["balanceamento"][nome]
            combinacoes = selecionar_combinacoes(nome, RANDOM_STATE + numero_modelo)
            print(f"\n{nome} | {estrategia} | {len(combinacoes)} configurações", flush=True)

            for numero_config, params in enumerate(combinacoes):
                config_id = f"{tarefa}_{nome}_{numero_config:02d}"
                params_json = json.dumps(params, sort_keys=True)
                print(f"  [{numero_config+1}/{len(combinacoes)}] {params_json}", flush=True)

                for numero_fold, X_tr, X_val, y_tr, y_val in cache[tarefa]:
                    # Classificador novo, sempre sem pesos internos.
                    modelo = criar_classificador(
                        nome=nome, n_classes=len(labels),
                        parametros=params, balanceamento="Sem",
                    )
                    inicio = perf_counter()
                    if estrategia == "Sem":
                        modelo.fit(X_tr, y_tr)
                    elif estrategia == "Pesos":
                        pesos = compute_sample_weight(class_weight="balanced", y=y_tr)
                        modelo.fit(X_tr, y_tr, sample_weight=pesos)
                    elif estrategia == "Oversampling":
                        X_over, y_over = oversampling(
                            X_tr, y_tr, seed=RANDOM_STATE + numero_fold,
                        )
                        modelo.fit(X_over, y_over)
                    else:
                        raise ValueError(f"Balanceamento desconhecido: {estrategia}")
                    fit_time = perf_counter() - inicio

                    inicio = perf_counter()
                    previsto = modelo.predict(X_val)
                    predict_time = perf_counter() - inicio

                    base = {
                        "tarefa": tarefa,
                        "modelo": nome,
                        "representacao": representacao,
                        "balanceamento": estrategia,
                        "config_id": config_id,
                        "baseline": numero_config == 0,
                        "parametros": params_json,
                        "fold": numero_fold,
                    }
                    metricas_folds.append({
                        **base,
                        "f1_macro": f1_score(y_val, previsto, labels=labels,
                                              average="macro", zero_division=0),
                        "precision_macro": precision_score(y_val, previsto, labels=labels,
                                                            average="macro", zero_division=0),
                        "recall_macro": recall_score(y_val, previsto, labels=labels,
                                                      average="macro", zero_division=0),
                        "accuracy": accuracy_score(y_val, previsto),
                        "balanced_accuracy": balanced_accuracy_score(y_val, previsto),
                        "fit_time_s": fit_time,
                        "predict_time_s": predict_time,
                    })
                    ps = precision_score(y_val, previsto, labels=labels,
                                         average=None, zero_division=0)
                    rs = recall_score(y_val, previsto, labels=labels,
                                      average=None, zero_division=0)
                    fs = f1_score(y_val, previsto, labels=labels,
                                  average=None, zero_division=0)
                    for i, classe in enumerate(nomes_classes):
                        metricas_classes.append({
                            "tarefa": tarefa, "modelo": nome,
                            "config_id": config_id, "fold": numero_fold,
                            "classe": classe,
                            "precision": ps[i], "recall": rs[i], "f1": fs[i],
                            "support": int(np.sum(y_val == i)),
                        })

    df_folds = pd.DataFrame(metricas_folds)
    df_classes = pd.DataFrame(metricas_classes)
    esperadas = sum(len(selecionar_combinacoes(nome, RANDOM_STATE + i))
                    for i, nome in enumerate(ordem_modelos)) * len(TAREFAS)
    if len(df_folds) != esperadas * N_FOLDS:
        raise AssertionError("Quantidade incorreta de ajustes")
    suporte_por_fold = df_classes.groupby(
        ["tarefa", "modelo", "config_id", "fold"]
    )["support"].sum()
    for (*_, numero_fold), suporte in suporte_por_fold.items():
        if suporte != len(folds[numero_fold - 1][1]):
            raise AssertionError(f"Suporte inconsistente no fold {numero_fold}")

    dimensoes = ["tarefa", "modelo", "representacao", "balanceamento",
                 "config_id", "baseline", "parametros"]
    resumo = df_folds.groupby(dimensoes)[list(METRICAS)].agg(["mean", "std"])
    resumo.columns = [f"{nome}_{estatistica}" for nome, estatistica in resumo.columns]
    resumo = resumo.reset_index()
    melhores = (resumo.sort_values("f1_macro_mean", ascending=False)
                .groupby(["tarefa", "modelo"], as_index=False).head(1)
                .sort_values(["tarefa", "modelo"]))

    df_folds.to_csv(SAIDA / "metricas_por_fold.csv", index=False)
    df_classes.to_csv(SAIDA / "metricas_por_classe.csv", index=False)
    resumo.to_csv(SAIDA / "todas_configuracoes.csv", index=False)
    melhores.to_csv(SAIDA / "melhores_configuracoes.csv", index=False)

    ellipsis = df_classes.merge(
        melhores[["tarefa", "modelo", "config_id"]],
        on=["tarefa", "modelo", "config_id"], how="inner", validate="many_to_one",
    )
    ellipsis = ellipsis[(ellipsis["tarefa"] == "Multiclasse") &
                        (ellipsis["classe"] == "Ellipsis")]
    ellipsis_resumo = ellipsis.groupby("modelo")[["precision", "recall", "f1"]].mean().reset_index()
    ellipsis_resumo.to_csv(SAIDA / "ellipsis_melhores.csv", index=False)

    verificar_referencia(resumo)
    print("\n========== MELHORES CONFIGURAÇÕES ==========", flush=True)
    print(melhores[["tarefa", "modelo", "representacao", "balanceamento",
                    "f1_macro_mean", "f1_macro_std", "parametros"]]
          .round(4).to_string(index=False))
    print("\n========== ELLIPSIS — MELHORES CONFIGURAÇÕES ==========", flush=True)
    print(ellipsis_resumo.round(3).to_string(index=False))
    print(f"\nTotal: {esperadas} configurações, {len(df_folds)} ajustes")
    print(f"Arquivos em: {SAIDA}")


if __name__ == "__main__":
    executar()
