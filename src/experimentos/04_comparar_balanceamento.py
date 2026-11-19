
from pathlib import Path
from time import perf_counter
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
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

from src.config import (
    DATASET_TREINO, LABELS_BINARIO, LABELS_MULTICLASSE,
    N_FOLDS, RANDOM_STATE, RESULTS_DIR,
)
from src.modelos import NOMES_MODELOS, criar_classificador, criar_representacao

SAIDA = RESULTS_DIR / "comparacao_balanceamento"
ESTRATEGIAS = ("Sem_Balanceamento", "Pesos", "Oversampling")
METRICAS = (
    "f1_macro", "precision_macro", "recall_macro",
    "accuracy", "balanced_accuracy", "fit_time_s", "predict_time_s",
)
DISTRIBUICAO_ESPERADA = {"Overlap": 185, "Truncation": 113, "Ellipsis": 20}


def carregar_dados():
    if not DATASET_TREINO.is_file():
        raise FileNotFoundError(f"Dataset não encontrado: {DATASET_TREINO}")
    df = pd.read_excel(DATASET_TREINO)
    obrigatorias = {"url", "bug type", "title", "body"}
    if not obrigatorias.issubset(df.columns):
        raise ValueError(f"Colunas ausentes: {sorted(obrigatorias - set(df.columns))}")
    if len(df) != 318 or df["bug type"].value_counts().to_dict() != DISTRIBUICAO_ESPERADA:
        raise ValueError("Quantidade ou distribuição de classes não corresponde à histórica")
    if df["url"].isna().any() or df["url"].duplicated().any():
        raise ValueError("URLs ausentes ou repetidas")
    textos = (
        df["title"].fillna("").astype(str) + " " +
        df["body"].fillna("").astype(str)
    ).str.strip().reset_index(drop=True)
    if textos.eq("").any():
        raise ValueError("Existem issues sem título e descrição")
    return df, textos


def oversampling_aleatorio(X_treino, y_treino, seed):
    """Reproduz o oversampling original, sem tocar nas amostras de validação."""
    rng = np.random.default_rng(seed)
    classes, quantidades = np.unique(y_treino, return_counts=True)
    maior = int(quantidades.max())
    indices = []
    for classe in classes:
        originais = np.flatnonzero(y_treino == classe)
        extras = rng.choice(originais, size=maior - len(originais), replace=True)
        indices.extend(originais.tolist())
        indices.extend(extras.tolist())
    indices = np.asarray(indices, dtype=int)
    rng.shuffle(indices)
    return X_treino[indices], y_treino[indices]


def calcular_metricas(y_real, y_prev, labels):
    return {
        "f1_macro": f1_score(y_real, y_prev, labels=labels,
                             average="macro", zero_division=0),
        "precision_macro": precision_score(y_real, y_prev, labels=labels,
                                           average="macro", zero_division=0),
        "recall_macro": recall_score(y_real, y_prev, labels=labels,
                                     average="macro", zero_division=0),
        "accuracy": accuracy_score(y_real, y_prev),
        "balanced_accuracy": balanced_accuracy_score(y_real, y_prev),
    }


def comparar_com_referencia(resumo):
    referencia = SAIDA / "referencia_resumo_balanceamento.csv"
    if not referencia.is_file():
        print("\nReferência histórica opcional não encontrada; comparação ignorada.")
        return
    antigo = pd.read_csv(referencia)
    chaves = ["tarefa", "representacao", "modelo", "estrategia"]
    comparado = resumo.merge(antigo, on=chaves, how="outer", indicator=True,
                             suffixes=("_novo", "_historico"), validate="one_to_one")
    if not comparado["_merge"].eq("both").all():
        raise AssertionError("Configurações não coincidem com a referência histórica")
    print("\n========== COMPARAÇÃO HISTÓRICA ==========")
    divergencias = []
    for metrica in METRICAS:
        if metrica.endswith("_time_s"):
            continue  # tempos não são reproduzíveis entre ambientes
        for estatistica in ("mean", "std"):
            coluna = f"{metrica}_{estatistica}"
            dif = (comparado[f"{coluna}_novo"] -
                   comparado[f"{coluna}_historico"]).abs().max()
            print(f"{coluna:30s} diferença máxima = {dif:.9f}")
            if dif > 1e-6:
                divergencias.append(coluna)
    if divergencias:
        raise AssertionError(f"Diferenças acima de 1e-6: {divergencias}")
    print("Reprodução histórica aprovada (tolerância de 1e-6).")


def executar():
    df, X = carregar_dados()
    y_multi = df["bug type"].map(LABELS_MULTICLASSE).to_numpy(dtype=int)
    y_bin = df["bug type"].map(LABELS_BINARIO).to_numpy(dtype=int)

    tarefas = {
        "Multiclasse": {
            "y": y_multi, "representacao": "Word",
            "labels": [0, 1, 2],
            "nomes": ["Overlap", "Truncation", "Ellipsis"],
        },
        "Binario": {
            "y": y_bin, "representacao": "Combined",
            "labels": [0, 1],
            "nomes": ["Overlap", "Truncation/Ellipsis"],
        },
    }

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(cv.split(X, y_multi))
    resultados_folds = []
    resultados_classes = []
    previsoes = []
    SAIDA.mkdir(parents=True, exist_ok=True)

    for tarefa, config in tarefas.items():
        y = config["y"]
        representacao = config["representacao"]
        labels = config["labels"]
        print(f"\n========== {tarefa} | {representacao} ==========", flush=True)

        for numero_fold, (indices_treino, indices_val) in enumerate(folds, 1):
            print(f"Fold {numero_fold}/{N_FOLDS}", flush=True)
            vetorizador = criar_representacao(representacao)
            # Ajustar representação somente com o treino do fold.
            X_tr = vetorizador.fit_transform(X.iloc[indices_treino])
            X_val = vetorizador.transform(X.iloc[indices_val])
            y_tr = y[indices_treino]
            y_val = y[indices_val]
            pesos = compute_sample_weight(class_weight="balanced", y=y_tr)
            X_over, y_over = oversampling_aleatorio(
                X_tr, y_tr, seed=RANDOM_STATE + numero_fold
            )

            for nome_modelo in NOMES_MODELOS:
                for estrategia in ESTRATEGIAS:
                    if nome_modelo == "KNN" and estrategia == "Pesos":
                        continue
                    print(f"  {nome_modelo} | {estrategia}", flush=True)
                    # Todos começam sem pesos internos: nunca aplicar pesos duas vezes.
                    modelo = criar_classificador(
                        nome_modelo, n_classes=len(labels), balanceamento="Sem"
                    )
                    inicio = perf_counter()
                    if estrategia == "Sem_Balanceamento":
                        modelo.fit(X_tr, y_tr)
                    elif estrategia == "Pesos":
                        modelo.fit(X_tr, y_tr, sample_weight=pesos)
                    else:
                        modelo.fit(X_over, y_over)
                    tempo_fit = perf_counter() - inicio

                    inicio = perf_counter()
                    y_prev = modelo.predict(X_val)
                    tempo_pred = perf_counter() - inicio
                    campos = {
                        "tarefa": tarefa, "representacao": representacao,
                        "modelo": nome_modelo, "estrategia": estrategia,
                        "fold": numero_fold,
                    }
                    resultados_folds.append({
                        **campos, **calcular_metricas(y_val, y_prev, labels),
                        "fit_time_s": tempo_fit,
                        "predict_time_s": tempo_pred,
                    })
                    prec = precision_score(y_val, y_prev, labels=labels,
                                           average=None, zero_division=0)
                    rec = recall_score(y_val, y_prev, labels=labels,
                                       average=None, zero_division=0)
                    f1s = f1_score(y_val, y_prev, labels=labels,
                                   average=None, zero_division=0)
                    for i, classe in enumerate(config["nomes"]):
                        resultados_classes.append({
                            **campos, "classe": classe,
                            "precision": float(prec[i]),
                            "recall": float(rec[i]),
                            "f1": float(f1s[i]),
                            "support": int(np.sum(y_val == labels[i])),
                        })
                    for indice, previsto in zip(indices_val, y_prev):
                        previsoes.append({
                            **campos, "url": df.iloc[indice]["url"],
                            "real": int(y[indice]), "previsto": int(previsto),
                        })

    df_folds = pd.DataFrame(resultados_folds)
    df_classes = pd.DataFrame(resultados_classes)
    df_prev = pd.DataFrame(previsoes)
    resumo = (
        df_folds.groupby(["tarefa", "representacao", "modelo", "estrategia"])
        [list(METRICAS)].agg(["mean", "std"])
    )
    resumo.columns = [f"{metrica}_{estatistica}" for metrica, estatistica in resumo.columns]
    resumo = resumo.reset_index().sort_values(
        ["tarefa", "f1_macro_mean"], ascending=[True, False]
    )

    # 14 configurações por tarefa (KNN sem pesos), 5 folds: 140 avaliações.
    if len(df_folds) != 140 or len(resumo) != 28:
        raise AssertionError("Número inesperado de configurações ou avaliações")
    if len(df_classes) != 350:
        raise AssertionError("Número inesperado de métricas por classe")
    if len(df_prev) != 28 * 318:
        raise AssertionError("Número inesperado de previsões")
    for chave, grupo in df_prev.groupby(["tarefa", "modelo", "estrategia"]):
        if len(grupo) != 318 or grupo["url"].nunique() != 318:
            raise AssertionError(f"Previsões faltando/duplicadas: {chave}")

    ellipsis = df_prev[df_prev["tarefa"] == "Multiclasse"]
    linhas_ellipsis = []
    for (modelo, estrategia), grupo in ellipsis.groupby(["modelo", "estrategia"]):
        real = grupo["real"].to_numpy()
        pred = grupo["previsto"].to_numpy()
        acertos = int(((real == 2) & (pred == 2)).sum())
        linhas_ellipsis.append({
            "modelo": modelo, "estrategia": estrategia,
            "total_ellipsis": int((real == 2).sum()),
            "acertos_ellipsis": acertos,
            "precision_ellipsis": precision_score(real == 2, pred == 2, zero_division=0),
            "recall_ellipsis": recall_score(real == 2, pred == 2, zero_division=0),
            "f1_ellipsis": f1_score(real == 2, pred == 2, zero_division=0),
        })
    df_ellipsis = pd.DataFrame(linhas_ellipsis)
    if not df_ellipsis["total_ellipsis"].eq(20).all():
        raise AssertionError("Número incorreto de exemplos Ellipsis")

    comparar_com_referencia(resumo)
    df_folds.to_csv(SAIDA / "metricas_por_fold.csv", index=False)
    df_classes.to_csv(SAIDA / "metricas_por_classe.csv", index=False)
    df_prev.to_csv(SAIDA / "previsoes_validacao.csv", index=False)
    resumo.to_csv(SAIDA / "resumo_balanceamento.csv", index=False)
    df_ellipsis.to_csv(SAIDA / "analise_ellipsis.csv", index=False)

    print("\n========== RESULTADOS FINAIS ==========")
    print(resumo[["tarefa", "modelo", "estrategia", "f1_macro_mean",
                  "f1_macro_std", "accuracy_mean"]].round(4).to_string(index=False))
    print("\n========== RECALL DE ELLIPSIS ==========")
    print(df_ellipsis.pivot(index="modelo", columns="estrategia",
                            values="recall_ellipsis").round(3).to_string())
    print("\nArquivos salvos em:", SAIDA)


if __name__ == "__main__":
    executar()
