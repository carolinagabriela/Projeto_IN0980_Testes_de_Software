
from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

from src.config import (
    DATASET_TREINO,
    LABELS_BINARIO,
    LABELS_MULTICLASSE,
    N_FOLDS,
    RANDOM_STATE,
    RESULTS_DIR,
)
from src.modelos import NOMES_MODELOS, criar_pipeline

SAIDA = RESULTS_DIR / "comparacao_entradas"
METRICAS = (
    "f1_macro", "precision_macro", "recall_macro", "accuracy", "balanced_accuracy"
)
CONTAGEM_ESPERADA = {"Overlap": 185, "Truncation": 113, "Ellipsis": 20}


def carregar_dados():
    if not DATASET_TREINO.is_file():
        raise FileNotFoundError(f"Treinamento não encontrado: {DATASET_TREINO}")
    df = pd.read_excel(DATASET_TREINO)
    obrigatorias = {"url", "bug type", "title", "body", "comments"}
    faltantes = obrigatorias.difference(df.columns)
    if faltantes:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(faltantes)}")
    if len(df) != 318:
        raise ValueError(f"Esperadas 318 issues de treino; encontradas {len(df)}")
    if df["url"].isna().any() or df["url"].duplicated().any():
        raise ValueError("URLs ausentes ou duplicadas no conjunto de treino")
    distribuicao = df["bug type"].value_counts().to_dict()
    if distribuicao != CONTAGEM_ESPERADA:
        raise ValueError(f"Classes inesperadas: {distribuicao}")
    for coluna in ("title", "body", "comments"):
        df[coluna] = df[coluna].fillna("").astype(str)
    entradas = {
        "T1_Title": df["title"].str.strip(),
        "T2_Title_Body": (df["title"] + " " + df["body"]).str.strip(),
        "T3_Title_Body_Comments": (
            df["title"] + " " + df["body"] + " " + df["comments"]
        ).str.strip(),
    }
    for nome, textos in entradas.items():
        if textos.eq("").any():
            raise ValueError(f"Existem textos vazios para {nome}")
    return df, entradas


def gerar_metricas(y_real, y_previsto, labels):
    return {
        "f1_macro": f1_score(
            y_real, y_previsto, labels=labels, average="macro", zero_division=0
        ),
        "precision_macro": precision_score(
            y_real, y_previsto, labels=labels, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_real, y_previsto, labels=labels, average="macro", zero_division=0
        ),
        "accuracy": accuracy_score(y_real, y_previsto),
        "balanced_accuracy": balanced_accuracy_score(y_real, y_previsto),
    }


def comparar_historico(resumo):
    """Compara com CSV histórico, se foi colocado com o nome de referência."""
    referencia = SAIDA / "referencia_resumo_comparacao.csv"
    if not referencia.is_file():
        print("\nReferência histórica opcional não encontrada; comparação ignorada.")
        return
    historico = pd.read_csv(referencia)
    chaves = ["entrada", "tarefa", "modelo"]
    if historico.duplicated(chaves).any() or resumo.duplicated(chaves).any():
        raise AssertionError("Chaves repetidas em resumo ou referência")
    comparado = resumo.merge(
        historico, on=chaves, how="outer", validate="one_to_one",
        suffixes=("_novo", "_historico"), indicator=True,
    )
    if not comparado["_merge"].eq("both").all():
        raise AssertionError("Conjunto de entradas/tarefas/modelos difere do histórico")
    nomes = [f"{metrica}_{estatistica}" for metrica in METRICAS
             for estatistica in ("mean", "std")]
    print("\n========== COMPARAÇÃO COM O HISTÓRICO ==========")
    divergencias = []
    for nome in nomes:
        diferenca = (comparado[f"{nome}_novo"] - comparado[f"{nome}_historico"]).abs()
        maximo = float(diferenca.max())
        print(f"{nome:30s} diferença máxima = {maximo:.9f}")
        if maximo > 1e-6:
            divergencias.append(nome)
    if divergencias:
        raise AssertionError(f"Métricas não reproduzidas (tolerância 1e-6): {divergencias}")
    print("Todos os indicadores reproduzidos dentro da tolerância de 1e-6.")


def executar():
    df, entradas = carregar_dados()
    SAIDA.mkdir(parents=True, exist_ok=True)
    y_multi = df["bug type"].map(LABELS_MULTICLASSE).to_numpy(dtype=int)
    y_binario = df["bug type"].map(LABELS_BINARIO).to_numpy(dtype=int)
    tarefas = {
        "Multiclasse": (y_multi, 3, [0, 1, 2]),
        "Binario": (y_binario, 2, [0, 1]),
    }

    # Folds definidos UMA vez pelas três classes, reutilizados pelas duas tarefas
    # e por todos os modelos e entradas. Sem acesso ao dataset_teste.xlsx.
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(cv.split(df, y_multi))
    resultados_folds, resultados_classes, previsoes = [], [], []
    total = len(entradas) * len(tarefas) * len(NOMES_MODELOS)
    contador = 0

    print("Treinamento:", len(df), "issues")
    print("Distribuição:", df["bug type"].value_counts().to_dict())
    print("Tarefas e entradas usam os mesmos folds estratificados por classe original.")
    for entrada, X in entradas.items():
        X = X.reset_index(drop=True)
        for tarefa, (y, n_classes, labels) in tarefas.items():
            for nome_modelo in NOMES_MODELOS:
                contador += 1
                print(f"[{contador:02d}/{total}] {entrada} | {tarefa} | {nome_modelo}",
                      flush=True)
                for fold, (indices_treino, indices_validacao) in enumerate(folds, 1):
                    pipeline = criar_pipeline(
                        nome=nome_modelo,
                        n_classes=n_classes,
                        representacao="Word",
                        balanceamento="Baseline",
                    )
                    # O TF-IDF é ajustado apenas com o treinamento do fold.
                    pipeline.fit(X.iloc[indices_treino], y[indices_treino])
                    y_real = y[indices_validacao]
                    y_prev = pipeline.predict(X.iloc[indices_validacao])
                    metricas = gerar_metricas(y_real, y_prev, labels)
                    resultados_folds.append({
                        "entrada": entrada, "tarefa": tarefa,
                        "modelo": nome_modelo, "fold": fold,
                        **metricas,
                    })
                    # Reproduz exatamente o formato histórico das métricas por classe.
                    for classe in labels:
                        resultados_classes.append({
                            "entrada": entrada, "tarefa": tarefa,
                            "modelo": nome_modelo, "fold": fold, "classe": classe,
                            "f1": f1_score(y_real, y_prev, labels=[classe],
                                           average="macro", zero_division=0),
                            "recall": recall_score(y_real, y_prev, labels=[classe],
                                                    average="macro", zero_division=0),
                        })
                    for indice, prev in zip(indices_validacao, y_prev):
                        previsoes.append({
                            "entrada": entrada, "tarefa": tarefa,
                            "modelo": nome_modelo, "fold": fold,
                            "url": df.iloc[indice]["url"],
                            "real": int(y[indice]), "previsto": int(prev),
                        })

    df_folds = pd.DataFrame(resultados_folds)
    df_classes = pd.DataFrame(resultados_classes)
    df_previsoes = pd.DataFrame(previsoes)
    resumo = (
        df_folds.groupby(["entrada", "tarefa", "modelo"])[list(METRICAS)]
        .agg(["mean", "std"])
    )
    resumo.columns = [f"{metrica}_{estatistica}"
                      for metrica, estatistica in resumo.columns]
    resumo = resumo.reset_index().sort_values(
        ["tarefa", "entrada", "f1_macro_mean"],
        ascending=[True, True, False],
    )

    # Verificações estruturais antes de escrever qualquer resultado.
    if len(df_folds) != 150 or len(df_classes) != 375 or len(df_previsoes) != 9540:
        raise AssertionError("Quantidade inesperada de métricas/previsões")
    for (_, _, _), grupo in df_previsoes.groupby(["entrada", "tarefa", "modelo"]):
        if len(grupo) != 318 or grupo["url"].nunique() != 318:
            raise AssertionError("Previsões fora dos folds ausentes ou duplicadas")

    # Primeiro verifica a referência; somente depois salva os CSVs novos.
    comparar_historico(resumo)
    df_folds.to_csv(SAIDA / "metricas_por_fold.csv", index=False)
    df_classes.to_csv(SAIDA / "metricas_por_classe.csv", index=False)
    df_previsoes.to_csv(SAIDA / "previsoes_validacao.csv", index=False)
    resumo.to_csv(SAIDA / "resumo_comparacao.csv", index=False)

    print("\n========== RESULTADO FINAL ==========")
    print(resumo[["entrada", "tarefa", "modelo", "f1_macro_mean", "f1_macro_std",
                  "accuracy_mean"]].round(4).to_string(index=False))
    print("\nArquivos salvos em:", SAIDA)


if __name__ == "__main__":
    executar()
