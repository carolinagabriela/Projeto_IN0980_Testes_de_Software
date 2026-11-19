
from pathlib import Path
import sys

# Permite executar o arquivo pelo caminho, a partir da raiz do repositório.
RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.config import DATASET_TREINO, N_FOLDS, RANDOM_STATE, RESULTS_DIR
from src.modelos import criar_pipeline

SAIDA = RESULTS_DIR / "baselines"
NOMES_MODELOS = ("Dummy", "DT", "RF", "SVM", "XGBoost", "KNN")
METRICAS = {
    "f1_macro": "f1_macro",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
}

MAPA_MULTI = {"Overlap": 0, "Truncation": 1, "Ellipsis": 2}
MAPA_BINARIO = {"Overlap": 0, "Truncation": 1, "Ellipsis": 1}


def carregar_treinamento():
    if not DATASET_TREINO.is_file():
        raise FileNotFoundError(f"Dataset de treinamento ausente: {DATASET_TREINO}")
    df = pd.read_excel(DATASET_TREINO)
    obrigatorias = {"url", "bug type", "title", "body"}
    ausentes = obrigatorias - set(df.columns)
    if ausentes:
        raise ValueError(f"Colunas ausentes: {sorted(ausentes)}")
    if len(df) != 318:
        raise ValueError(f"Esperados 318 registros de treino; encontrados {len(df)}")
    if df["url"].isna().any() or df["url"].duplicated().any():
        raise ValueError("URLs ausentes ou duplicadas no treino")
    contagem = df["bug type"].value_counts().to_dict()
    esperada = {"Overlap": 185, "Truncation": 113, "Ellipsis": 20}
    if contagem != esperada:
        raise ValueError(f"Distribuição inesperada: {contagem}; esperada: {esperada}")

    # Preserva a construção exata do texto no experimento original.
    for coluna in ("title", "body"):
        df[coluna] = df[coluna].fillna("").astype(str)
    texto = (df["title"] + " " + df["body"]).str.strip()
    if texto.eq("").any():
        raise ValueError("Existem issues com texto de entrada vazio")
    return df, texto


def executar():
    df, X = carregar_treinamento()
    SAIDA.mkdir(parents=True, exist_ok=True)

    tarefas = {
        "Multiclasse": (df["bug type"].map(MAPA_MULTI).astype(int), 3),
        "Binario": (df["bug type"].map(MAPA_BINARIO).astype(int), 2),
    }

    resultados = []
    resultados_folds = []
    for tarefa, (y, n_classes) in tarefas.items():
        # Reprodução histórica: cada tarefa tem sua própria estratificação.
        cv = StratifiedKFold(
            n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE
        )
        print(f"\n========== {tarefa} ==========", flush=True)

        for nome in NOMES_MODELOS:
            print(f"Treinando {nome}...", flush=True)
            pipeline = criar_pipeline(
                nome=nome,
                n_classes=n_classes,
                representacao="Word",
                balanceamento="Baseline",
            )
            scores = cross_validate(
                pipeline, X, y, cv=cv, scoring=METRICAS,
                n_jobs=1, error_score="raise",
            )
            resultado = {
                "problema": tarefa,
                "modelo": nome,
                "f1_macro": float(np.mean(scores["test_f1_macro"])),
                "f1_std": float(np.std(scores["test_f1_macro"], ddof=0)),
                "precision_macro": float(np.mean(scores["test_precision_macro"])),
                "recall_macro": float(np.mean(scores["test_recall_macro"])),
                "accuracy": float(np.mean(scores["test_accuracy"])),
                "balanced_accuracy": float(np.mean(scores["test_balanced_accuracy"])),
                "fit_time_s": float(np.mean(scores["fit_time"])),
            }
            resultados.append(resultado)

            for indice_fold in range(N_FOLDS):
                resultados_folds.append({
                    "problema": tarefa,
                    "modelo": nome,
                    "fold": indice_fold + 1,
                    **{
                        metrica: float(scores[f"test_{metrica}"][indice_fold])
                        for metrica in METRICAS
                    },
                    "fit_time_s": float(scores["fit_time"][indice_fold]),
                })
            print(f"  F1-macro: {resultado['f1_macro']:.4f}", flush=True)

    resumo = pd.DataFrame(resultados).sort_values(
        ["problema", "f1_macro"], ascending=[True, False]
    )
    folds_df = pd.DataFrame(resultados_folds)
    resumo.to_csv(SAIDA / "resultados_validacao.csv", index=False)
    folds_df.to_csv(SAIDA / "metricas_por_fold.csv", index=False)

    print("\n========== RESULTADOS ==========")
    print(resumo[["problema", "modelo", "f1_macro", "f1_std", "accuracy"]]
          .round(4).to_string(index=False))
    print(f"\nArquivos gerados em: {SAIDA}")

    # Comparação opcional com CSV histórico copiado à pasta baselines.
    referencia = SAIDA / "referencia_resultados_validacao.csv"
    if referencia.is_file():
        historico = pd.read_csv(referencia)
        chaves = ["problema", "modelo"]
        comparacao = resumo.merge(
            historico, on=chaves, suffixes=("_novo", "_historico"),
            validate="one_to_one", indicator=True,
        )
        if len(comparacao) != len(resumo) or not comparacao["_merge"].eq("both").all():
            raise AssertionError("Modelos/tarefas não correspondem ao histórico")
        metricas_comparacao = [
            "f1_macro", "f1_std", "precision_macro", "recall_macro",
            "accuracy", "balanced_accuracy",
        ]
        print("\n========== COMPARAÇÃO COM HISTÓRICO ==========")
        houve_diferenca = False
        for metrica in metricas_comparacao:
            diferencas = (
                comparacao[f"{metrica}_novo"] -
                comparacao[f"{metrica}_historico"]
            ).abs()
            maior = float(diferencas.max())
            print(f"{metrica}: maior diferença = {maior:.8f}")
            if maior > 1e-6:
                houve_diferenca = True
        if houve_diferenca:
            print("ATENÇÃO: métricas divergentes. Conferir versões e configurações.")
        else:
            print("Métricas reproduzidas dentro da tolerância de 1e-6.")
        print("Tempos de execução não são comparados.")


if __name__ == "__main__":
    executar()
