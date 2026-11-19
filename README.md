# Classificação textual de falhas de internacionalização e localização

Projeto da disciplina da pós graduação *IN0980 - Tópicos Avançados em Engenharia de Software 3* sobre a classificação automática de falhas de interface associadas à **internacionalização e localização (i18n/l10n)**, a partir de informações textuais de *issues* do GitHub. São utilizados algoritmos tradicionais de aprendizado de máquina e representações TF-IDF, **sem análise de screenshots e sem uso de modelos de linguagem**.

O estudo compara a identificação de três categorias individualmente e uma tarefa binária com duas categorias agrupadas.

## Tarefas e modelos

- **Multiclasse:** `Overlap`, `Truncation` e `Ellipsis`.
- **Binária:** `Overlap` versus `Truncation/Ellipsis`.

Durante o desenvolvimento, foram comparados **Decision Tree (DT), Random Forest (RF), SVM linear, XGBoost e KNN**, além de um `DummyClassifier` como referência inicial. As comparações investigaram campos textuais, representações TF-IDF, técnicas de balanceamento e ajuste limitado de hiperparâmetros.

Os modelos selecionados para a avaliação final foram:

| Tarefa | Classificador | Representação | Balanceamento |
|---|---|---|---|
| Multiclasse | XGBoost | Word | Sem balanceamento |
| Multiclasse | Decision Tree | Word | Pesos por classe |
| Binária | XGBoost | Combined | Sem balanceamento |
| Binária | SVM linear | Combined | Pesos por classe |

As configurações completas estão em `src/modelos.py`, na constante `CONFIGURACOES_FINAIS`.

## Dados

O conjunto original contém **6.410 registros**. Após selecionar as três categorias de interesse, recuperar os textos da coleta histórica, tratar URLs repetidas, rótulos conflitantes e duplicatas textuais, o conjunto experimental ficou com **398 issues**.

| Classe | Treinamento | Teste | Total |
|---|---:|---:|---:|
| Overlap | 185 | 47 | 232 |
| Truncation | 113 | 28 | 141 |
| Ellipsis | 20 | 5 | 25 |
| **Total** | **318** | **80** | **398** |

Os experimentos de desenvolvimento utilizam apenas as **318 issues de treinamento**, com validação cruzada estratificada de cinco folds. As **80 issues reservadas** são utilizadas na avaliação final.

> **Reprodução da coleta:** a preparação usa `data/snapshots/github_issues_original.xlsx`, que preserva as respostas históricas do GitHub. `02_enriquecer.py` não faz novas requisições à API. O snapshot deve estar disponível para reproduzir o processamento; uma coleta atual poderia retornar textos diferentes.

## Estrutura do repositório

```text
.
├── data/
│   ├── raw/                 # Dataset original
│   ├── snapshots/           # Snapshot da coleta histórica do GitHub
│   ├── processed/           # Datasets gerados, incluindo treino e teste
│   └── review/              # Relatórios de duplicatas, conflitos e falhas
├── src/
│   ├── config.py            # Caminhos, sementes e mapeamento de classes
│   ├── modelos.py           # Classificadores e representações TF-IDF
│   ├── preparacao/          # Filtragem, enriquecimento, limpeza e divisão
│   ├── experimentos/        # Experimentos 1 a 5
│   └── avaliacao/           # Avaliação final e consolidação
├── results/                 # Métricas, previsões, tabelas e figuras
├── docs/                    # Material complementar do projeto
├── requirements.txt
└── README.md
```

## Resultados finais

| Tarefa | Modelo | Accuracy | F1-macro |
|---|---|---:|---:|
| Multiclasse | XGBoost | 0,5875 | **0,3786** |
| Multiclasse | Decision Tree | 0,5125 | 0,3451 |
| Binária | XGBoost | **0,6875** | 0,6593 |
| Binária | SVM | 0,6750 | **0,6615** |

Na avaliação multiclasse, nenhum dos dois modelos selecionados identificou corretamente as **cinco issues de Ellipsis** do teste. Esse resultado deve ser interpretado com cautela devido ao número reduzido de exemplos.

Também foi feita uma comparação do XGBoost sobre os **mesmos rótulos binários**:

| Abordagem | F1-macro |
|---|---:|
| Previsões do XGBoost multiclasse agrupadas posteriormente | 0,6107 |
| XGBoost treinado diretamente para a tarefa binária | **0,6593** |

As duas configurações diferem também em representação textual e hiperparâmetros; por isso, a diferença observada não isola somente o efeito do agrupamento de classes.

## Reprodutibilidade e limitações

- O dataset é desbalanceado, especialmente para `Ellipsis` (20 registros de treinamento e 5 de teste).
- As categorias descrevem manifestações visuais, mas os modelos recebem apenas texto. Este trabalho não demonstra isoladamente que a ausência de screenshots seja a causa dos erros.
- Os resultados de validação foram utilizados para selecionar configurações. O conjunto de teste foi reservado para a avaliação final; reproduzir seus resultados posteriormente não constitui uma nova avaliação independente.
- A inclusão de comentários, no Experimento 2, utiliza o histórico textual recuperado da issue e não necessariamente o texto disponível no momento da abertura.


