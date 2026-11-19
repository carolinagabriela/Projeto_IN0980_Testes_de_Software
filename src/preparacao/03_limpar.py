
import argparse
import re
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
ENTRADA = BASE_DIR / 'data/processed/dataset_filtrado_enriquecido.xlsx'
PASTA_DADOS = BASE_DIR / 'data/processed'
PASTA_REVISAO = BASE_DIR / 'data/review'

COLUNAS_NECESSARIAS = {'url', 'bug type', 'fetch_status', 'title', 'body', 'comments'}
MAPA_CLASSES = {
    'overlap': 'Overlap',
    'truncation': 'Truncation',
    'elipsis': 'Ellipsis',
    'ellipsis': 'Ellipsis',
}


def normalizar_texto(valor):
    """Mantém a normalização original: minúsculas e espaços colapsados."""
    return re.sub(r'\s+', ' ', str(valor).lower()).strip()


def comparar_arquivos_existentes(caminho, dados):
    """Permite reutilizar relatórios históricos se forem logicamente idênticos."""
    if not caminho.exists():
        return False
    existente = (pd.read_csv(caminho) if caminho.suffix == '.csv'
                 else pd.read_excel(caminho))
    # Para relatórios originais, comparamos valores sem depender do dtype inferido.
    if list(existente.columns) != list(dados.columns) or len(existente) != len(dados):
        return False
    return existente.fillna('').astype(str).reset_index(drop=True).equals(
        dados.fillna('').astype(str).reset_index(drop=True)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sobrescrever', action='store_true',
                        help='Permite substituir saídas já existentes após conferência.')
    args = parser.parse_args()

    if not ENTRADA.is_file():
        raise FileNotFoundError(f'Arquivo de entrada não encontrado: {ENTRADA}')

    df = pd.read_excel(ENTRADA)
    ausentes = COLUNAS_NECESSARIAS - set(df.columns)
    if ausentes:
        raise ValueError(f'Colunas obrigatórias ausentes: {sorted(ausentes)}')
    if len(df) != 620:
        raise ValueError(f'Esperadas 620 ocorrências; encontradas {len(df)}.')

    # Reproduzir a primeira limpeza do projeto antigo: sem mudar a ordem.
    df_ok = df.loc[df['fetch_status'] == 'ok'].copy()
    falhas = df.loc[df['fetch_status'] != 'ok'].copy()
    if len(df_ok) != 616 or len(falhas) != 4:
        raise ValueError(f'Status históricos inesperados: ok={len(df_ok)}, falhas={len(falhas)}')

    classes_normalizadas = df_ok['bug type'].astype('string').str.strip().str.lower()
    df_ok['bug type'] = classes_normalizadas.map(MAPA_CLASSES)
    if df_ok['bug type'].isna().any():
        raise ValueError('Foram encontrados rótulos desconhecidos.')

    df_ok['url_normalizada'] = (
        df_ok['url'].astype('string').str.strip().str.rstrip('/').str.lower()
    )
    if df_ok['url_normalizada'].isna().any() or (df_ok['url_normalizada'] == '').any():
        raise ValueError('Foram encontradas issues com URL vazia.')

    qtd_classes = df_ok.groupby('url_normalizada')['bug type'].nunique()
    urls_conflitantes = qtd_classes[qtd_classes > 1].index
    conflitos_urls = df_ok.loc[
        df_ok['url_normalizada'].isin(urls_conflitantes)
    ].copy()
    if len(urls_conflitantes) != 23 or len(conflitos_urls) != 107:
        raise ValueError('O número histórico de URLs/rótulos conflitantes não foi reproduzido.')

    sem_conflitos = df_ok.loc[
        ~df_ok['url_normalizada'].isin(urls_conflitantes)
    ].copy()
    duplicatas_urls = sem_conflitos.loc[
        sem_conflitos.duplicated(subset=['url_normalizada'], keep='first')
    ].copy()
    urls_limpas = sem_conflitos.drop_duplicates(
        subset=['url_normalizada'], keep='first'
    ).copy()
    if len(duplicatas_urls) != 107 or len(urls_limpas) != 402:
        raise ValueError('A deduplicação por URL não reproduziu 107 exclusões e 402 issues.')

    for coluna in ['title', 'body', 'comments']:
        urls_limpas[coluna] = urls_limpas[coluna].fillna('').astype(str)

    # O projeto antigo só fez esta segunda limpeza no script dividir_dataset.py.
    # Aqui antecipamos a operação: o futuro 04_dividir.py deve APENAS dividir.
    urls_limpas['_texto_comparacao'] = (
        urls_limpas['title'].apply(normalizar_texto)
        + ' '
        + urls_limpas['body'].apply(normalizar_texto)
    )
    rotulos_por_texto = urls_limpas.groupby('_texto_comparacao')['bug type'].nunique()
    textos_conflitantes = rotulos_por_texto[rotulos_por_texto > 1].index
    conflitos_textuais = urls_limpas.loc[
        urls_limpas['_texto_comparacao'].isin(textos_conflitantes)
    ].copy()
    sem_conflitos_textuais = urls_limpas.loc[
        ~urls_limpas['_texto_comparacao'].isin(textos_conflitantes)
    ].copy()
    duplicatas_textuais = sem_conflitos_textuais.loc[
        sem_conflitos_textuais.duplicated(subset=['_texto_comparacao'], keep='first')
    ].copy()
    dataset_limpo = sem_conflitos_textuais.drop_duplicates(
        subset=['_texto_comparacao'], keep='first'
    ).copy()

    if len(conflitos_textuais) != 2 or len(duplicatas_textuais) != 2 or len(dataset_limpo) != 398:
        raise ValueError('A limpeza textual não reproduziu 2 conflitos, 2 duplicatas e 398 issues.')

    def sem_auxiliar(frame):
        return frame.drop(columns=['url_normalizada', '_texto_comparacao'], errors='ignore').reset_index(drop=True)

    urls_limpas_saida = sem_auxiliar(urls_limpas)
    dataset_limpo_saida = sem_auxiliar(dataset_limpo)
    conflitos_urls_saida = sem_auxiliar(conflitos_urls)
    conflitos_textuais_saida = sem_auxiliar(conflitos_textuais)
    duplicatas_urls_saida = sem_auxiliar(duplicatas_urls)
    duplicatas_textuais_saida = sem_auxiliar(duplicatas_textuais)

    esperado = {'Overlap': 232, 'Truncation': 141, 'Ellipsis': 25}
    if dataset_limpo_saida['bug type'].value_counts().to_dict() != esperado:
        raise ValueError('Distribuição final incompatível com o projeto original.')
    if dataset_limpo_saida['url'].duplicated().any():
        raise ValueError('Persistem URLs repetidas após a limpeza.')

    # Verificação adicional contra os conjuntos históricos, se estiverem presentes.
    referencia_treino = PASTA_DADOS / 'dataset_treino.xlsx'
    referencia_teste = PASTA_DADOS / 'dataset_teste.xlsx'
    if referencia_treino.is_file() and referencia_teste.is_file():
        referencia = pd.concat(
            [pd.read_excel(referencia_treino), pd.read_excel(referencia_teste)],
            ignore_index=True,
        )
        a = dataset_limpo_saida.set_index('url')['bug type'].sort_index()
        b = referencia.set_index('url')['bug type'].sort_index()
        if not a.equals(b):
            raise ValueError('URLs ou rótulos diferem dos conjuntos históricos de treino/teste.')
        print('Referências de treino/teste: 398 URLs e classes idênticas.')

    resumo = pd.DataFrame([
        ('Entrada enriquecida', 620),
        ('Falhas de coleta', len(falhas)),
        ('Recuperações bem-sucedidas', len(df_ok)),
        ('URLs conflitantes (distintas)', len(urls_conflitantes)),
        ('Ocorrências excluídas por conflitos de URL', len(conflitos_urls)),
        ('Duplicatas de URL excluídas', len(duplicatas_urls)),
        ('Issues únicas após limpeza por URL', len(urls_limpas_saida)),
        ('Ocorrências excluídas por conflitos textuais', len(conflitos_textuais)),
        ('Duplicatas textuais excluídas', len(duplicatas_textuais)),
        ('Dataset final para divisão', len(dataset_limpo_saida)),
    ], columns=['etapa', 'quantidade'])

    saidas = {
        PASTA_DADOS / 'dataset_urls_limpas.xlsx': urls_limpas_saida,
        PASTA_DADOS / 'dataset_limpo.xlsx': dataset_limpo_saida,
        PASTA_REVISAO / 'issues_conflitantes_reproduzidos.xlsx': conflitos_urls_saida,
        PASTA_REVISAO / 'conflitos_textuais_reproduzidos.xlsx': conflitos_textuais_saida,
        PASTA_REVISAO / 'duplicatas_urls_removidas.xlsx': duplicatas_urls_saida,
        PASTA_REVISAO / 'duplicatas_textuais_removidas.xlsx': duplicatas_textuais_saida,
        PASTA_REVISAO / 'resumo_limpeza.csv': resumo,
    }
    for caminho, dados in saidas.items():
        if caminho.exists() and not args.sobrescrever and not comparar_arquivos_existentes(caminho, dados):
            raise FileExistsError(
                f'O arquivo já existe e difere do resultado calculado: {caminho}\n'
                'Revise-o antes de usar --sobrescrever.'
            )

    for caminho, dados in saidas.items():
        caminho.parent.mkdir(parents=True, exist_ok=True)
        if caminho.exists() and not args.sobrescrever:
            print(f'Relatório existente preservado: {caminho.name}')
            continue
        if caminho.suffix == '.csv':
            dados.to_csv(caminho, index=False)
        else:
            dados.to_excel(caminho, index=False, engine='openpyxl')

    print('\n=== LIMPEZA CONCLUÍDA ===')
    print(resumo.to_string(index=False))
    print('\nDistribuição final:')
    print(dataset_limpo_saida['bug type'].value_counts().to_string())
    print(f'\nSaída final: {PASTA_DADOS / "dataset_limpo.xlsx"}')
    print('Próximo script: 04_dividir.py (não repetir limpeza textual).')


if __name__ == '__main__':
    main()
