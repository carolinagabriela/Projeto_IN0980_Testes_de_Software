
from pathlib import Path
import json
import sys

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.utils.class_weight import compute_sample_weight

from src.config import (
    DATASET_TREINO, DATASET_TESTE, RESULTS_DIR,
    LABELS_MULTICLASSE, LABELS_BINARIO,
)
from src.modelos import CONFIGURACOES_FINAIS, criar_pipeline

SAIDA = RESULTS_DIR / "avaliacao_final"
NOMES_CLASSES = {
    "Multiclasse": ["Overlap", "Truncation", "Ellipsis"],
    "Binario": ["Overlap", "Truncation/Ellipsis"],
}
MAPAS = {"Multiclasse": LABELS_MULTICLASSE, "Binario": LABELS_BINARIO}
DISTRIBUICAO_TREINO = {"Overlap": 185, "Truncation": 113, "Ellipsis": 20}
DISTRIBUICAO_TESTE = {"Overlap": 47, "Truncation": 28, "Ellipsis": 5}
ESPERADOS = {
    ("Multiclasse", "XGBoost"), ("Multiclasse", "DT"),
    ("Binario", "XGBoost"), ("Binario", "SVM"),
}


def carregar_dataset(caminho, total_esperado, distribuicao_esperada):
    if not caminho.is_file():
        raise FileNotFoundError(f"Dataset não encontrado: {caminho}")
    df = pd.read_excel(caminho)
    obrigatorias = {"url", "bug type", "title", "body"}
    if not obrigatorias.issubset(df.columns):
        raise ValueError(f"Colunas ausentes em {caminho.name}: {sorted(obrigatorias-set(df.columns))}")
    if len(df) != total_esperado or df["bug type"].value_counts().to_dict() != distribuicao_esperada:
        raise ValueError(f"Quantidade ou distribuição diferente da histórica: {caminho.name}")
    if df["url"].isna().any():
        raise ValueError(f"Há URLs vazias em {caminho.name}")
    urls = df["url"].astype(str).str.strip().str.lower()
    if urls.duplicated().any():
        raise ValueError(f"Há URLs repetidas em {caminho.name}")
    for coluna in ("title", "body"):
        df[coluna] = df[coluna].fillna("").astype(str)
    texto = (df["title"] + " " + df["body"]).str.strip()
    if texto.eq("").any():
        raise ValueError(f"Há issues sem título e descrição em {caminho.name}")
    return df.reset_index(drop=True), texto.reset_index(drop=True), urls


def conferir_separacao(treino, teste, urls_treino, urls_teste):
    if set(urls_treino) & set(urls_teste):
        raise ValueError("Vazamento: URLs presentes no treinamento e no teste")

    def textos_normalizados(df):
        return set(
            (df["title"] + " " + df["body"])
            .str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
        )

    if textos_normalizados(treino) & textos_normalizados(teste):
        raise ValueError("Vazamento: title+body idênticos entre treino e teste")


def calcular_metricas(y_real, y_prev, nomes):
    labels = list(range(len(nomes)))
    return {
        "accuracy": accuracy_score(y_real, y_prev),
        "balanced_accuracy": balanced_accuracy_score(y_real, y_prev),
        "precision_macro": precision_score(y_real, y_prev, labels=labels, average="macro", zero_division=0),
        "recall_macro": recall_score(y_real, y_prev, labels=labels, average="macro", zero_division=0),
        "f1_macro": f1_score(y_real, y_prev, labels=labels, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_real, y_prev, labels=labels, average="weighted", zero_division=0),
    }


def registrar_resultados(tarefa, modelo, tipo, y_real, y_prev, nomes,
                         metricas, por_classe, matrizes):
    chave = {"tarefa": tarefa, "modelo": modelo, "tipo_avaliacao": tipo}
    metricas.append({**chave, **calcular_metricas(y_real, y_prev, nomes)})
    relatorio = classification_report(
        y_real, y_prev, labels=list(range(len(nomes))),
        target_names=nomes, output_dict=True, zero_division=0,
    )
    for classe in nomes:
        m = relatorio[classe]
        por_classe.append({
            **chave, "classe": classe,
            "precision": m["precision"], "recall": m["recall"],
            "f1": m["f1-score"], "support": int(m["support"]),
        })
    matriz = confusion_matrix(y_real, y_prev, labels=list(range(len(nomes))))
    for i, real in enumerate(nomes):
        for j, previsto in enumerate(nomes):
            matrizes.append({
                **chave, "classe_real": real,
                "classe_prevista": previsto, "quantidade": int(matriz[i, j]),
            })
    return matriz


def salvar_grafico_matriz(matriz, nomes, titulo, caminho):
    fig, ax = plt.subplots(figsize=(7, 5))
    imagem = ax.imshow(matriz, cmap="Blues")
    ax.set_xticks(range(len(nomes)), labels=nomes, rotation=20, ha="right")
    ax.set_yticks(range(len(nomes)), labels=nomes)
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    ax.set_title(titulo)
    for i in range(len(nomes)):
        for j in range(len(nomes)):
            valor = matriz[i, j]
            ax.text(j, i, str(valor), ha="center", va="center",
                    color="white" if valor > matriz.max() / 2 else "black")
    fig.colorbar(imagem, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(caminho, dpi=180)
    plt.close(fig)


def verificar_configuracoes():
    chaves = [(c["tarefa"], c["modelo"]) for c in CONFIGURACOES_FINAIS]
    if len(chaves) != 4 or set(chaves) != ESPERADOS:
        raise ValueError("CONFIGURACOES_FINAIS não contém os quatro modelos esperados")
    if any(c["balanceamento"] not in ("Sem", "Pesos") for c in CONFIGURACOES_FINAIS):
        raise ValueError("Balanceamento inesperado na avaliação final")

    # Confere que a seleção do Experimento 5 corresponde às configurações congeladas.
    referencia = RESULTS_DIR / "ajuste_hiperparametros" / "melhores_configuracoes.csv"
    if not referencia.is_file():
        print("Aviso: melhores_configuracoes.csv não encontrado; usando CONFIGURACOES_FINAIS congeladas.")
        return
    melhores = pd.read_csv(referencia)
    colunas = {"tarefa", "modelo", "representacao", "balanceamento", "parametros"}
    if not colunas.issubset(melhores.columns):
        raise ValueError("CSV do Experimento 5 não possui colunas de configuração exigidas")
    for config in CONFIGURACOES_FINAIS:
        linhas = melhores.loc[
            (melhores["tarefa"] == config["tarefa"]) &
            (melhores["modelo"] == config["modelo"])
        ]
        if len(linhas) != 1:
            raise ValueError(f"Configuração selecionada ausente ou duplicada: {config['tarefa']} {config['modelo']}")
        linha = linhas.iloc[0]
        if (linha["representacao"] != config["representacao"] or
                linha["balanceamento"] != config["balanceamento"] or
                json.loads(linha["parametros"]) != config["parametros"]):
            raise ValueError(f"Divergência do Experimento 5: {config['tarefa']} {config['modelo']}")
    print("Configurações finais conferidas com o Experimento 5: 4/4 OK.")


def main():
    verificar_configuracoes()
    treino, texto_treino, urls_treino = carregar_dataset(
        DATASET_TREINO, 318, DISTRIBUICAO_TREINO
    )
    teste, texto_teste, urls_teste = carregar_dataset(
        DATASET_TESTE, 80, DISTRIBUICAO_TESTE
    )
    conferir_separacao(treino, teste, urls_treino, urls_teste)
    print("Datasets conferidos: 318 treino, 80 teste, sem URLs ou textos idênticos compartilhados.")

    metricas, por_classe, matrizes, previsoes = [], [], [], []
    predicoes_por_modelo = {}
    SAIDA.mkdir(parents=True, exist_ok=True)

    for config in CONFIGURACOES_FINAIS:
        tarefa, nome_modelo = config["tarefa"], config["modelo"]
        nomes = NOMES_CLASSES[tarefa]
        mapa = MAPAS[tarefa]
        y_treino = treino["bug type"].map(mapa).to_numpy(dtype=int)
        y_teste = teste["bug type"].map(mapa).to_numpy(dtype=int)
        pipeline = criar_pipeline(
            nome=nome_modelo,
            n_classes=len(nomes),
            representacao=config["representacao"],
            parametros=config["parametros"],
            balanceamento=config["balanceamento"],
        )
        kwargs = {}
        if config["balanceamento"] == "Pesos":
            kwargs["classifier__sample_weight"] = compute_sample_weight(
                class_weight="balanced", y=y_treino
            )

        print(f"\n=== {tarefa} | {nome_modelo} ===", flush=True)
        pipeline.fit(texto_treino, y_treino, **kwargs)
        y_prev = np.asarray(pipeline.predict(texto_teste), dtype=int)
        predicoes_por_modelo[(tarefa, nome_modelo)] = y_prev

        for i, previsto in enumerate(y_prev):
            previsoes.append({
                "url": teste.loc[i, "url"],
                "tarefa": tarefa,
                "modelo": nome_modelo,
                "classe_original": teste.loc[i, "bug type"],
                "real": int(y_teste[i]),
                "previsto": int(previsto),
                "real_nome": nomes[y_teste[i]],
                "previsto_nome": nomes[previsto],
                "correto": bool(y_teste[i] == previsto),
            })

        matriz = registrar_resultados(
            tarefa, nome_modelo, "original", y_teste, y_prev, nomes,
            metricas, por_classe, matrizes,
        )
        salvar_grafico_matriz(
            matriz, nomes, f"{tarefa} — {nome_modelo}",
            SAIDA / f"matriz_{tarefa.lower()}_{nome_modelo.lower()}.png",
        )
        print(pd.Series(calcular_metricas(y_teste, y_prev, nomes)).round(4).to_string())
        print(classification_report(
            y_teste, y_prev, labels=list(range(len(nomes))),
            target_names=nomes, zero_division=0,
        ))
        print("Matriz de confusão:\n", matriz)

    # Mesma tarefa binária e mesmas 80 issues, para as duas abordagens XGBoost.
    y_bin_real = teste["bug type"].map(LABELS_BINARIO).to_numpy(dtype=int)
    multi_agrupado = np.where(predicoes_por_modelo[("Multiclasse", "XGBoost")] == 0, 0, 1)
    bin_direto = predicoes_por_modelo[("Binario", "XGBoost")]
    comum = pd.DataFrame([
        {"abordagem": "XGBoost_multiclasse_agrupado", **calcular_metricas(y_bin_real, multi_agrupado, NOMES_CLASSES["Binario"])},
        {"abordagem": "XGBoost_binario_direto", **calcular_metricas(y_bin_real, bin_direto, NOMES_CLASSES["Binario"])},
    ])

    pd.DataFrame(metricas).to_csv(SAIDA / "metricas_gerais.csv", index=False)
    pd.DataFrame(por_classe).to_csv(SAIDA / "metricas_por_classe.csv", index=False)
    pd.DataFrame(matrizes).to_csv(SAIDA / "matrizes_confusao.csv", index=False)
    pd.DataFrame(previsoes).to_csv(SAIDA / "previsoes_teste.csv", index=False)
    comum.to_csv(SAIDA / "comparacao_binaria_comum.csv", index=False)
    with (SAIDA / "configuracoes_finais.json").open("w", encoding="utf-8") as arq:
        json.dump(CONFIGURACOES_FINAIS, arq, ensure_ascii=False, indent=2)

    if len(previsoes) != 4 * len(teste) or len(metricas) != 4:
        raise AssertionError("Quantidade inesperada de métricas ou previsões")
    # Comparação opcional da reprodução com as previsões históricas.
    # Copie a versão antiga para results/avaliacao_final/referencia_previsoes_teste.csv.
    ref_path = SAIDA / "referencia_previsoes_teste.csv"
    if ref_path.is_file():
        historicas = pd.read_csv(ref_path)
        atuais = pd.DataFrame(previsoes)
        colunas_chave = ["tarefa", "modelo", "url"]
        base = historicas[colunas_chave + ["real", "previsto"]].merge(
            atuais[colunas_chave + ["real", "previsto"]],
            on=colunas_chave, how="outer", suffixes=("_historico", "_atual"),
            validate="one_to_one", indicator=True,
        )
        diferentes = base.loc[
            (base["_merge"] != "both") |
            (base["real_historico"] != base["real_atual"]) |
            (base["previsto_historico"] != base["previsto_atual"])
        ]
        diferentes.to_csv(SAIDA / "divergencias_historicas.csv", index=False)
        print(f"\nComparação opcional com histórico: {len(diferentes)} diferenças de previsão/registro.")
        if len(diferentes):
            print(diferentes[["tarefa", "modelo", "url", "previsto_historico", "previsto_atual"]].to_string(index=False))
    else:
        print("\nReferência de previsões históricas não encontrada; conferência opcional ignorada.")

    print("\n=== COMPARAÇÃO BINÁRIA COMUM — XGBoost ===")
    print(comum.round(4).to_string(index=False))
    print(f"\nArquivos salvos em: {SAIDA}")
    print("Avaliação reproduzida. Não use os resultados de teste para ajustar os modelos.")


if __name__ == "__main__":
    main()
