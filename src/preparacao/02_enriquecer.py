
from pathlib import Path
import argparse

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
FILTRADO = BASE_DIR / "data" / "processed" / "dataset_filtrado.xlsx"
SNAPSHOT = BASE_DIR / "data" / "snapshots" / "github_issues_original.xlsx"
SAIDA = BASE_DIR / "data" / "processed" / "dataset_filtrado_enriquecido.xlsx"
RELATORIO_ERROS = BASE_DIR / "data" / "review" / "falhas_coleta.csv"

COLUNAS_ENRIQUECIMENTO = [
    "body", "comments", "comments_count", "fetch_status", "fetch_error"
]

def conferir_estrutura(filtrado: pd.DataFrame, snapshot: pd.DataFrame) -> None:
    if "url" not in filtrado or "url" not in snapshot:
        raise ValueError("A coluna 'url' deve existir nos dois arquivos.")
    if "bug type" not in filtrado:
        raise ValueError("Coluna 'bug type' ausente no dataset filtrado.")

    faltantes = [c for c in COLUNAS_ENRIQUECIMENTO if c not in snapshot]
    if faltantes:
        raise ValueError(f"Snapshot sem as colunas: {faltantes}")

    if len(filtrado) != 620 or len(snapshot) != 620:
        raise ValueError(
            "Esperávamos 620 ocorrências em ambos os arquivos. "
            f"Filtrado: {len(filtrado)}; snapshot: {len(snapshot)}."
        )

    # URLs não são identificadores únicos neste dataset: a mesma issue aparece
    # várias vezes, às vezes associada a rótulos diferentes. Por isso, juntar
    # (merge) apenas pela URL multiplicaria registros e alteraria a amostra.
    if not filtrado["url"].fillna("").astype(str).reset_index(drop=True).equals(
        snapshot["url"].fillna("").astype(str).reset_index(drop=True)
    ):
        raise ValueError(
            "As URLs ou a ordem das ocorrências diferem entre filtrado "
            "e snapshot. Não é seguro associar os textos automaticamente."
        )

    # Confirma também que todas as colunas históricas de origem coincidem.
    # Isso protege contra permutações de linhas com a mesma URL e diferenças
    # de rótulos, títulos e demais metadados.
    faltantes_origem = [col for col in filtrado.columns if col not in snapshot]
    if faltantes_origem:
        raise ValueError(f"Snapshot sem colunas de origem: {faltantes_origem}")

    pd.testing.assert_frame_equal(
        filtrado.reset_index(drop=True),
        snapshot[filtrado.columns].reset_index(drop=True),
        check_dtype=False,
        check_like=False,
        obj="Metadados originais (filtrado vs. snapshot)",
    )

    distribuicao = filtrado["bug type"].value_counts().to_dict()
    esperado = {"Overlap": 344, "Truncation": 226, "Elipsis": 50}
    if distribuicao != esperado:
        raise ValueError(
            f"Distribuição diferente da histórica: {distribuicao}; "
            f"esperada: {esperado}"
        )

    status = snapshot["fetch_status"].fillna("sem_status").value_counts().to_dict()
    if status != {"ok": 616, "error": 4}:
        raise ValueError(
            f"Status de coleta divergente: {status}; "
            "esperado: 616 'ok' e 4 'error'."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduzir enriquecimento usando snapshot histórico, sem API."
    )
    parser.add_argument(
        "--sobrescrever", action="store_true",
        help="Permitir substituir um arquivo de saída já existente.",
    )
    args = parser.parse_args()

    for caminho in (FILTRADO, SNAPSHOT):
        if not caminho.is_file():
            raise FileNotFoundError(f"Arquivo obrigatório não encontrado: {caminho}")

    if SAIDA.exists() and not args.sobrescrever:
        raise FileExistsError(
            f"Saída já existe: {SAIDA}. Para regenerar, execute com "
            "--sobrescrever depois de verificar o arquivo."
        )

    filtrado = pd.read_excel(FILTRADO)
    snapshot = pd.read_excel(SNAPSHOT)
    conferir_estrutura(filtrado, snapshot)

    # Combina os metadados da filtragem atual com os textos e resultados da
    # coleta histórica. A associação é posicional APÓS verificar igualdade
    # completa de URLs, ordem, rótulos e demais metadados.
    enriquecido = filtrado.reset_index(drop=True).copy()
    for coluna in COLUNAS_ENRIQUECIMENTO:
        enriquecido[coluna] = snapshot[coluna].reset_index(drop=True)

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    RELATORIO_ERROS.parent.mkdir(parents=True, exist_ok=True)

    # Cria a saída apenas depois de todas as verificações passarem.
    temporario = SAIDA.with_name("dataset_filtrado_enriquecido.tmp.xlsx")
    try:
        enriquecido.to_excel(temporario, index=False, engine="openpyxl")
        temporario.replace(SAIDA)
    finally:
        if temporario.exists():
            temporario.unlink()

    falhas = enriquecido.loc[
        enriquecido["fetch_status"].eq("error"),
        ["url", "bug type", "fetch_status", "fetch_error"],
    ]
    falhas.to_csv(RELATORIO_ERROS, index=False, encoding="utf-8-sig")

    print("\n=== ENRIQUECIMENTO HISTÓRICO REPRODUZIDO ===")
    print(f"Ocorrências filtradas: {len(filtrado)}")
    print(f"Ocorrências no snapshot: {len(snapshot)}")
    print("URLs, ordem e metadados originais: idênticos")
    print("\nStatus da coleta histórica:")
    print(enriquecido["fetch_status"].value_counts().to_string())
    print(f"\nArquivo enriquecido: {SAIDA}")
    print(f"Relatório das quatro falhas: {RELATORIO_ERROS}")
    print("\nAs falhas serão excluídas no próximo script de limpeza.")
    print("Nenhuma chamada à API do GitHub foi realizada.")


if __name__ == "__main__":
    main()
