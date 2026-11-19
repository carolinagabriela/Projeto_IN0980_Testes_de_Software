
from pathlib import Path
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

from src.config import (
    DATASET_TREINO, LABELS_BINARIO, LABELS_MULTICLASSE,
    N_FOLDS, RANDOM_STATE, RESULTS_DIR,
)
from src.modelos import NOMES_MODELOS, criar_pipeline

SAIDA = RESULTS_DIR / "comparacao_representacoes"
REPRESENTACOES = ("R1_Word", "R2_Character", "R3_Combined")
METRICAS = ("f1_macro", "precision_macro", "recall_macro", "accuracy", "balanced_accuracy")
DISTRIBUICAO_ESPERADA = {"Overlap": 185, "Truncation": 113, "Ellipsis": 20}


def carregar_dados():
    if not DATASET_TREINO.is_file():
        raise FileNotFoundError(f"Dataset não encontrado: {DATASET_TREINO}")
    df = pd.read_excel(DATASET_TREINO)
    obrigatorias = {"url", "bug type", "title", "body"}
    if not obrigatorias.issubset(df.columns):
        raise ValueError(f"Colunas ausentes: {sorted(obrigatorias - set(df.columns))}")
    if len(df) != 318:
        raise ValueError(f"Esperadas 318 issues; encontradas {len(df)}")
    if df["url"].isna().any() or df["url"].duplicated().any():
        raise ValueError("URLs ausentes ou duplicadas no treinamento")
    if df["bug type"].value_counts().to_dict() != DISTRIBUICAO_ESPERADA:
        raise ValueError("Distribuição de classes diferente da histórica")
    titulo = df["title"].fillna("").astype(str)
    descricao = df["body"].fillna("").astype(str)
    textos = (titulo + " " + descricao).str.strip().reset_index(drop=True)
    if textos.eq("").any():
        raise ValueError("Existem registros sem texto em title + body")
    return df, textos


def metricas_gerais(y_real, y_pred, labels):
    return {
        "f1_macro": f1_score(y_real, y_pred, labels=labels, average="macro", zero_division=0),
        "precision_macro": precision_score(y_real, y_pred, labels=labels, average="macro", zero_division=0),
        "recall_macro": recall_score(y_real, y_pred, labels=labels, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_real, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_real, y_pred),
    }


def comparar_com_referencia(resumo):
    referencia = SAIDA / "referencia_resumo_representacoes.csv"
    if not referencia.is_file():
        print("\nReferência histórica opcional não encontrada; comparação ignorada.")
        return
    historico = pd.read_csv(referencia)
    chaves = ["representacao", "tarefa", "modelo"]
    if historico.duplicated(chaves).any() or resumo.duplicated(chaves).any():
        raise AssertionError("Chaves duplicadas no resumo ou na referência")
    compara = resumo.merge(historico, on=chaves, how="outer", indicator=True,
                           suffixes=("_novo", "_historico"), validate="one_to_one")
    if not compara["_merge"].eq("both").all():
        raise AssertionError("Representações, tarefas ou modelos diferem da referência")
    print("\n========== COMPARAÇÃO HISTÓRICA ==========")
    divergencias = []
    for metrica in METRICAS:
        for estatistica in ("mean", "std"):
            coluna = f"{metrica}_{estatistica}"
            delta = (compara[f"{coluna}_novo"] - compara[f"{coluna}_historico"]).abs()
            maior = float(delta.max())
            print(f"{coluna:30s} diferença máxima = {maior:.9f}")
            if maior > 1e-6:
                divergencias.append(coluna)
    if divergencias:
        raise AssertionError(f"Divergências superiores a 1e-6: {divergencias}")
    print("Reprodução histórica aprovada (tolerância: 1e-6).")


def executar():
    df, X = carregar_dados()
    y_multi = df["bug type"].map(LABELS_MULTICLASSE).to_numpy(dtype=int)
    y_bin = df["bug type"].map(LABELS_BINARIO).to_numpy(dtype=int)
    tarefas = {
        "Multiclasse": (y_multi, 3, [0, 1, 2]),
        "Binario": (y_bin, 2, [0, 1]),
    }
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    # Os folds são definidos UMA vez usando as três classes originais.
    folds = list(cv.split(X, y_multi))
    SAIDA.mkdir(parents=True, exist_ok=True)

    resultados_folds = []
    resultados_classes = []
    previsoes = []
    total = len(REPRESENTACOES) * len(tarefas) * len(NOMES_MODELOS)
    contador = 0
    print(f"Treinamento: {len(df)} issues | folds fixos: {N_FOLDS}")
    print("Distribuição:", df["bug type"].value_counts().to_dict())

    for representacao in REPRESENTACOES:
        for tarefa, (y, n_classes, labels) in tarefas.items():
            for modelo in NOMES_MODELOS:
                contador += 1
                print(f"[{contador:02d}/{total}] {representacao} | {tarefa} | {modelo}",
                      flush=True)
                for fold, (indices_treino, indices_val) in enumerate(folds, 1):
                    pipeline = criar_pipeline(
                        nome=modelo, n_classes=n_classes,
                        representacao=representacao, balanceamento="Baseline",
                    )
                    # Ajusta Word/Character/Combined SOMENTE nos textos de treino.
                    pipeline.fit(X.iloc[indices_treino], y[indices_treino])
                    y_real = y[indices_val]
                    y_pred = pipeline.predict(X.iloc[indices_val])
                    resultados_folds.append({
                        "representacao": representacao, "tarefa": tarefa,
                        "modelo": modelo, "fold": fold,
                        **metricas_gerais(y_real, y_pred, labels),
                    })
                    precisions = precision_score(y_real, y_pred, labels=labels,
                                                 average=None, zero_division=0)
                    recalls = recall_score(y_real, y_pred, labels=labels,
                                            average=None, zero_division=0)
                    f1s = f1_score(y_real, y_pred, labels=labels,
                                   average=None, zero_division=0)
                    for i, classe in enumerate(labels):
                        resultados_classes.append({
                            "representacao": representacao, "tarefa": tarefa,
                            "modelo": modelo, "fold": fold, "classe": classe,
                            "precision": float(precisions[i]),
                            "recall": float(recalls[i]), "f1": float(f1s[i]),
                            "support": int(np.sum(y_real == classe)),
                        })
                    for indice, previsto in zip(indices_val, y_pred):
                        previsoes.append({
                            "representacao": representacao, "tarefa": tarefa,
                            "modelo": modelo, "fold": fold,
                            "url": df.iloc[indice]["url"],
                            "real": int(y[indice]), "previsto": int(previsto),
                        })

    df_folds = pd.DataFrame(resultados_folds)
    df_classes = pd.DataFrame(resultados_classes)
    df_prev = pd.DataFrame(previsoes)
    resumo = (df_folds.groupby(["representacao", "tarefa", "modelo"])[list(METRICAS)]
              .agg(["mean", "std"]))
    resumo.columns = [f"{metrica}_{estatistica}"
                      for metrica, estatistica in resumo.columns]
    resumo = resumo.reset_index().sort_values(
        ["tarefa", "representacao", "f1_macro_mean"],
        ascending=[True, True, False],
    )

    # 30 configurações, cinco folds, 150 avaliações. Total 318 previsões
    # fora dos folds para cada configuração.
    assert len(df_folds) == 150, f"Quantidade de folds: {len(df_folds)}"
    assert len(df_classes) == 375, f"Quantidade de classes: {len(df_classes)}"
    assert len(df_prev) == 9540, f"Quantidade de previsões: {len(df_prev)}"
    for chave, grupo in df_prev.groupby(["representacao", "tarefa", "modelo"]):
        if len(grupo) != 318 or grupo["url"].nunique() != 318:
            raise AssertionError(f"Previsões ausentes/repetidas em {chave}")

    # Reporta Ellipsis a partir das previsões fora dos folds (20 registros),
    # não da média das taxas dos folds. As duas podem diferir para F1.
    ellipsis = []
    for (rep, modelo), grupo in df_prev[df_prev["tarefa"] == "Multiclasse"].groupby(
        ["representacao", "modelo"]
    ):
        real = grupo["real"].to_numpy()
        previsto = grupo["previsto"].to_numpy()
        acertos = int(((real == 2) & (previsto == 2)).sum())
        ellipsis.append({
            "representacao": rep, "modelo": modelo,
            "acertos_ellipsis": acertos, "total_ellipsis": int((real == 2).sum()),
            "precision_ellipsis": precision_score(real == 2, previsto == 2,
                                                    zero_division=0),
            "recall_ellipsis": recall_score(real == 2, previsto == 2,
                                             zero_division=0),
            "f1_ellipsis": f1_score(real == 2, previsto == 2, zero_division=0),
        })
    df_ellipsis = pd.DataFrame(ellipsis)
    if not df_ellipsis["total_ellipsis"].eq(20).all():
        raise AssertionError("Ellipsis deveria ter 20 registros por configuração")

    # Se houver a referência histórica, valida antes de sobrescrever resultados.
    comparar_com_referencia(resumo)

    df_folds.to_csv(SAIDA / "metricas_por_fold.csv", index=False)
    df_classes.to_csv(SAIDA / "metricas_por_classe.csv", index=False)
    df_prev.to_csv(SAIDA / "previsoes_validacao.csv", index=False)
    resumo.to_csv(SAIDA / "resumo_representacoes.csv", index=False)
    df_ellipsis.to_csv(SAIDA / "analise_ellipsis.csv", index=False)

    print("\n========== RESULTADO FINAL ==========")
    print(resumo[["representacao", "tarefa", "modelo", "f1_macro_mean",
                  "f1_macro_std", "accuracy_mean"]].round(4).to_string(index=False))
    print("\n========== ELLIPSIS (20 ISSUES) ==========")
    print(df_ellipsis[["representacao", "modelo", "acertos_ellipsis",
                        "recall_ellipsis", "f1_ellipsis"]].round(3).to_string(index=False))
    print("\nArquivos salvos em:", SAIDA)


if __name__ == "__main__":
    executar()
