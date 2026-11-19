# Consolidação dos experimentos — i18n/l10n

## Protocolo da avaliação final
- 318 issues de desenvolvimento; 80 issues reservadas para teste.
- Quatro configurações selecionadas previamente por validação cruzada.
- A consolidação lê os CSVs existentes: não treina, não seleciona e não altera modelos.
- A comparação multiclasse vs. binária direta usa métricas de tarefas diferentes; para um confronto controlado em rótulos, usar `tabela_03_comparacao_xgboost_binario.csv`.
- O teste possui somente cinco registros de Ellipsis: relatar contagens absolutas e evitar conclusões fortes.

## Métricas gerais — teste

| tarefa | modelo | accuracy | balanced_accuracy | precision_macro | recall_macro | f1_macro |
| --- | --- | --- | --- | --- | --- | --- |
| Binario | SVM | 0.6750 | 0.6602 | 0.6636 | 0.6602 | 0.6615 |
| Binario | XGBoost | 0.6875 | 0.6573 | 0.6815 | 0.6573 | 0.6593 |
| Multiclasse | XGBoost | 0.5875 | 0.3863 | 0.3750 | 0.3863 | 0.3786 |
| Multiclasse | DT | 0.5125 | 0.3341 | 0.3570 | 0.3341 | 0.3451 |

## Comparação comum (XGBoost)

| abordagem | accuracy | balanced_accuracy | f1_macro |
| --- | --- | --- | --- |
| XGBoost_multiclasse_agrupado | 0.6375 | 0.6103 | 0.6107 |
| XGBoost_binario_direto | 0.6875 | 0.6573 | 0.6593 |

## Verificações
- 29/29 verificações passaram.

## Arquivos anteriores encontrados
- baseline: 12 linhas; OK
- entradas: 30 linhas; OK
- representacoes: 30 linhas; OK
- balanceamento: 28 linhas; OK
- hiperparametros: 10 linhas; OK

## Avisos
- Nenhum aviso.

## Revisão metodológica posterior
- Conferir coleta e filtros iniciais; duplicatas de URL e texto; separação estratificada; pipelines e comparabilidade de folds; seleção de configurações; ausência de vazamento; arquivos redundantes/intermediários.
- Não modificar a seleção dos modelos em função do teste independente.
