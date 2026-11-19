
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
ARQUIVO_ENTRADA = BASE_DIR / "data" / "raw" / "valid_dataset.xlsx"
ARQUIVO_SAIDA = BASE_DIR / "data" / "processed" / "dataset_filtrado.xlsx"

MAPA_ROTULOS = {
    "overlap": "Overlap",
    "truncation": "Truncation",
    "elipsis": "Elipsis",  
}
CONTAGENS_ESPERADAS = {
    "Overlap": 344,
    "Truncation": 226,
    "Elipsis": 50,
}

def main():
    if not ARQUIVO_ENTRADA.is_file():
        raise FileNotFoundError(f"Dataset original não encontrado: {ARQUIVO_ENTRADA}")

    df = pd.read_excel(ARQUIVO_ENTRADA)
    colunas_obrigatorias = {"bug type", "url"}
    ausentes = colunas_obrigatorias - set(df.columns)
    if ausentes:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(ausentes)}")

    print(f"Registros do dataset original: {len(df)}")
    if len(df) != 6410:
        raise ValueError("O dataset de entrada não possui os 6.410 registros esperados.")

    # A normalização serve apenas para reconhecer rótulos com espaços ou
    # capitalização diferentes. Registros multirrótulo não são selecionados.
    rotulos = df["bug type"].astype("string").str.strip().str.lower()
    selecionados = rotulos.isin(MAPA_ROTULOS)

    filtrado = df.loc[selecionados].copy()
    filtrado["bug type"] = rotulos.loc[selecionados].map(MAPA_ROTULOS)

    contagens = filtrado["bug type"].value_counts().to_dict()
    if contagens != CONTAGENS_ESPERADAS:
        raise ValueError(
            "Distribuição diferente da referência histórica. "
            f"Esperado: {CONTAGENS_ESPERADAS}. Obtido: {contagens}."
        )

    if filtrado["url"].isna().any():
        raise ValueError("Há URLs ausentes na seleção.")

    ARQUIVO_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    filtrado.to_excel(ARQUIVO_SAIDA, index=False)

    print(f"Registros selecionados: {len(filtrado)}")
    print("Distribuição por classe:")
    for classe, total in CONTAGENS_ESPERADAS.items():
        print(f"  {classe}: {total}")
    print(f"Arquivo gerado: {ARQUIVO_SAIDA}")
    print("Atenção: duplicatas e rótulos conflitantes serão tratados posteriormente.")

if __name__ == "__main__":
    main()
