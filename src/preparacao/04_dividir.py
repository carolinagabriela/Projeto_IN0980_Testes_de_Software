
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[2]
PASTA = BASE_DIR / "data" / "processed"
ENTRADA = PASTA / "dataset_limpo.xlsx"
SAIDA_TREINO = PASTA / "dataset_treino.xlsx"
SAIDA_TESTE = PASTA / "dataset_teste.xlsx"

SEED = 42
TEST_SIZE = 0.20
CONTAGENS = {"Overlap": 232, "Truncation": 141, "Ellipsis": 25}
CONTAGENS_TREINO = {"Overlap": 185, "Truncation": 113, "Ellipsis": 20}
CONTAGENS_TESTE = {"Overlap": 47, "Truncation": 28, "Ellipsis": 5}


def verificar_distribuicao(df: pd.DataFrame, esperada: dict, nome: str) -> None:
    obtida = df["bug type"].value_counts().to_dict()
    if obtida != esperada:
        raise ValueError(f"Distribuição inesperada em {nome}: {obtida}; esperada: {esperada}")


def comparar_conteudo(caminho: Path, gerado: pd.DataFrame) -> None:
    """Compara colunas, ordem das linhas e valores, tratando NaN e '' como vazio.

    A comparação como texto neutraliza diferenças de inferência de dtypes ao
    carregar planilhas sem permitir mudanças no conteúdo dos registros.
    """
    historico = pd.read_excel(caminho)
    if list(historico.columns) != list(gerado.columns):
        raise ValueError(f"As colunas do arquivo histórico diferem: {caminho}")
    if len(historico) != len(gerado):
        raise ValueError(f"O tamanho do arquivo histórico difere: {caminho}")

    a = historico.reset_index(drop=True).fillna("").astype(str)
    b = gerado.reset_index(drop=True).fillna("").astype(str)
    if not a.equals(b):
        urls_a = historico["url"].astype(str).tolist()
        urls_b = gerado["url"].astype(str).tolist()
        primeira = next((i for i, (x, y) in enumerate(zip(urls_a, urls_b)) if x != y), None)
        detalhe = (f"Primeira URL divergente na linha {primeira + 2}."
                   if primeira is not None else "URLs iguais, mas outros campos divergem.")
        raise ValueError(
            f"O arquivo histórico difere do conjunto reproduzido: {caminho}\n"
            f"{detalhe} O arquivo existente foi preservado."
        )


def main() -> None:
    if not ENTRADA.is_file():
        raise FileNotFoundError(
            f"Dataset limpo não encontrado: {ENTRADA}\n"
            "Execute primeiro src/preparacao/03_limpar.py."
        )

    df = pd.read_excel(ENTRADA)
    obrigatorias = {"url", "bug type", "title", "body", "comments"}
    ausentes = obrigatorias - set(df.columns)
    if ausentes:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(ausentes)}")
    if len(df) != 398:
        raise ValueError(f"Esperadas 398 issues; encontradas {len(df)}.")
    verificar_distribuicao(df, CONTAGENS, "dataset limpo")
    if df["url"].isna().any() or df["url"].duplicated().any():
        raise ValueError("Existem URLs vazias ou repetidas no dataset limpo.")

    # No fluxo antigo, esses três campos foram preenchidos com string vazia
    # antes da divisão. A ordem original das linhas é preservada.
    for coluna in ["title", "body", "comments"]:
        df[coluna] = df[coluna].fillna("").astype(str)

    # A limpeza textual já ocorreu em 03_limpar.py; aqui fazemos apenas
    # verificações de integridade para evitar vazamento por texto idêntico.
    import re
    def normalizar(texto: str) -> str:
        return re.sub(r"\s+", " ", str(texto).lower()).strip()

    assinatura = df["title"].map(normalizar) + " " + df["body"].map(normalizar)
    if assinatura.duplicated().any():
        raise ValueError("Persistem textos normalizados duplicados no dataset limpo.")

    # Não ordenar nem embaralhar previamente: a ordem é parte da reprodução.
    treino, teste = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=df["bug type"],
    )
    treino = treino.reset_index(drop=True)
    teste = teste.reset_index(drop=True)

    if len(treino) != 318 or len(teste) != 80:
        raise ValueError("A divisão não reproduziu 318 registros de treino e 80 de teste.")
    verificar_distribuicao(treino, CONTAGENS_TREINO, "treinamento")
    verificar_distribuicao(teste, CONTAGENS_TESTE, "teste")

    urls_treino, urls_teste = set(treino["url"]), set(teste["url"])
    if urls_treino & urls_teste:
        raise ValueError("Vazamento: URLs presentes tanto em treino quanto em teste.")
    if urls_treino | urls_teste != set(df["url"]):
        raise ValueError("A união de treino e teste não corresponde ao dataset limpo.")

    assinatura_por_url = dict(zip(df["url"], assinatura))
    textos_treino = {assinatura_por_url[url] for url in urls_treino}
    textos_teste = {assinatura_por_url[url] for url in urls_teste}
    if textos_treino & textos_teste:
        raise ValueError("Vazamento: textos idênticos nos dois conjuntos.")

    # Primeiro compara TODOS os arquivos que já existam, sem escrever nada.
    for caminho, dados in [(SAIDA_TREINO, treino), (SAIDA_TESTE, teste)]:
        if caminho.exists():
            comparar_conteudo(caminho, dados)

    # Se existirem e forem idênticos, preserva os arquivos históricos;
    # caso contrário, cria as planilhas reprodutíveis.
    for caminho, dados in [(SAIDA_TREINO, treino), (SAIDA_TESTE, teste)]:
        if caminho.exists():
            print(f"Histórico idêntico; arquivo preservado: {caminho.name}")
        else:
            caminho.parent.mkdir(parents=True, exist_ok=True)
            dados.to_excel(caminho, index=False, engine="openpyxl")
            print(f"Arquivo criado: {caminho.name}")

    print("\n=== DIVISÃO ESTRATIFICADA CONCLUÍDA ===")
    print(f"Dataset limpo: {len(df)} issues")
    print(f"Treinamento: {len(treino)} issues")
    print(treino["bug type"].value_counts().to_string())
    print(f"\nTeste: {len(teste)} issues")
    print(teste["bug type"].value_counts().to_string())
    print("\nURLs repetidas entre conjuntos: 0")
    print("Textos idênticos entre conjuntos: 0")
    print("Teste histórico preservado e divisão reproduzível.")


if __name__ == "__main__":
    main()
