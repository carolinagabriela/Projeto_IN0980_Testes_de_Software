
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[2]
RESULTS = BASE / "results"
FINAL = RESULTS / "avaliacao_final"
OUT = RESULTS / "consolidacao"
OUT.mkdir(parents=True, exist_ok=True)

ARQUIVOS = {
    "final_geral": FINAL / "metricas_gerais.csv",
    "final_classes": FINAL / "metricas_por_classe.csv",
    "final_matrizes": FINAL / "matrizes_confusao.csv",
    "final_previsoes": FINAL / "previsoes_teste.csv",
    "final_comum": FINAL / "comparacao_binaria_comum.csv",
    "baseline": RESULTS / "baselines" / "resultados_validacao.csv",
    "entradas": RESULTS / "comparacao_entradas" / "resumo_comparacao.csv",
    "representacoes": RESULTS / "comparacao_representacoes" / "resumo_representacoes.csv",
    "balanceamento": RESULTS / "comparacao_balanceamento" / "resumo_balanceamento.csv",
    "hiperparametros": RESULTS / "ajuste_hiperparametros" / "melhores_configuracoes.csv",
}

warnings = []
def ler(chave, obrigatorio=False):
    path = ARQUIVOS[chave]
    if not path.is_file():
        msg = f"Arquivo {'obrigatório' if obrigatorio else 'opcional'} ausente: {path.relative_to(BASE)}"
        if obrigatorio:
            raise FileNotFoundError(msg)
        warnings.append(msg)
        return None
    return pd.read_csv(path)

def exigir(df, cols, nome):
    faltam = set(cols) - set(df.columns)
    if faltam:
        raise ValueError(f"{nome}: colunas ausentes: {sorted(faltam)}")

def salvar_csv(df, nome):
    df.to_csv(OUT / nome, index=False, float_format="%.6f")

def fig_save(nome):
    plt.tight_layout()
    plt.savefig(OUT / nome, dpi=200, bbox_inches="tight")
    plt.close()

def grafico_barras(df, coluna_x, coluna_y, titulo, arquivo, xlabel="", limite=None):
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bars = ax.barh(df[coluna_x], df[coluna_y])
    ax.invert_yaxis()
    ax.set_title(titulo)
    ax.set_xlabel(xlabel or coluna_y)
    if limite is not None:
        ax.set_xlim(0, limite)
    else:
        ax.set_xlim(0, max(0.1, df[coluna_y].max() * 1.17))
    for b, v in zip(bars, df[coluna_y]):
        ax.text(b.get_width() + 0.008, b.get_y() + b.get_height()/2, f"{v:.3f}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=.20)
    ax.set_axisbelow(True)
    fig_save(arquivo)

# ====================== FONTES FINAIS ======================
geral = ler("final_geral", obrigatorio=True)
classes = ler("final_classes", obrigatorio=True)
matrizes = ler("final_matrizes", obrigatorio=True)
previsoes = ler("final_previsoes", obrigatorio=True)
comum = ler("final_comum", obrigatorio=True)
exigir(geral, ["tarefa", "modelo", "tipo_avaliacao", "accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"], "metricas_gerais")
exigir(classes, ["tarefa", "modelo", "tipo_avaliacao", "classe", "precision", "recall", "f1", "support"], "metricas_por_classe")
exigir(matrizes, ["tarefa", "modelo", "tipo_avaliacao", "classe_real", "classe_prevista", "quantidade"], "matrizes_confusao")
exigir(previsoes, ["tarefa", "modelo", "url", "real", "previsto"], "previsoes_teste")
exigir(comum, ["abordagem", "f1_macro", "accuracy", "balanced_accuracy", "precision_macro", "recall_macro"], "comparacao_binaria_comum")

final = geral.loc[geral.tipo_avaliacao.eq("original")].copy()
selecoes = {("Multiclasse", "XGBoost"), ("Multiclasse", "DT"), ("Binario", "XGBoost"), ("Binario", "SVM")}
atuais = set(zip(final.tarefa, final.modelo))
if atuais != selecoes or len(final) != 4:
    raise ValueError(f"Configurações finais inesperadas. Encontradas: {sorted(atuais)}")
final["chave"] = final["tarefa"] + " | " + final["modelo"]
final = final.sort_values(["tarefa", "f1_macro"], ascending=[True, False])
salvar_csv(final, "tabela_01_metricas_finais.csv")

classes = classes.loc[classes.tipo_avaliacao.eq("original")].copy()
classes["chave"] = classes["tarefa"] + " | " + classes["modelo"]
salvar_csv(classes.sort_values(["tarefa", "modelo", "classe"]), "tabela_02_metricas_por_classe.csv")
salvar_csv(comum, "tabela_03_comparacao_xgboost_binario.csv")

for tarefa in ("Multiclasse", "Binario"):
    subset = final.loc[final.tarefa.eq(tarefa)]
    grafico_barras(subset, "modelo", "f1_macro", f"F1-macro no teste — {tarefa}", f"fig_01_f1_{tarefa.lower()}.png", "F1-macro", .8)

for tarefa in ("Multiclasse", "Binario"):
    subset = classes.loc[classes.tarefa.eq(tarefa)]
    ordem = ["Overlap", "Truncation", "Ellipsis"] if tarefa == "Multiclasse" else ["Overlap", "Truncation/Ellipsis"]
    dados = subset.pivot(index="classe", columns="modelo", values="f1").reindex(ordem)
    ax = dados.plot(kind="bar", figsize=(8, 4.7), rot=0, ylim=(0, .85))
    ax.set_title(f"F1 por classe — {tarefa}")
    ax.set_xlabel("")
    ax.set_ylabel("F1")
    ax.legend(title="Modelo", frameon=False)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    fig_save(f"fig_02_f1_classes_{tarefa.lower()}.png")

comparar = comum.copy()
comparar["abordagem_curta"] = comparar["abordagem"].replace({
    "XGBoost_multiclasse_agrupado": "Multiclasse agrupada",
    "XGBoost_binario_direto": "Binário direto"
})
grafico_barras(comparar, "abordagem_curta", "f1_macro", "XGBoost: duas formas de resolver a tarefa binária", "fig_03_comparacao_binaria_comum.png", "F1-macro", .8)

# Matrizes de confusão: mesma fonte numérica do script de avaliação.
for tarefa, modelo in sorted(selecoes):
    m = matrizes.loc[(matrizes.tarefa == tarefa) & (matrizes.modelo == modelo) & (matrizes.tipo_avaliacao == "original")]
    nomes = ["Overlap", "Truncation", "Ellipsis"] if tarefa == "Multiclasse" else ["Overlap", "Truncation/Ellipsis"]
    pivot = m.pivot(index="classe_real", columns="classe_prevista", values="quantidade").reindex(index=nomes, columns=nomes)
    if pivot.isna().any().any():
        raise ValueError(f"Matriz incompleta: {tarefa}/{modelo}")
    arr = pivot.to_numpy(dtype=int)
    if arr.sum() != 80:
        raise ValueError(f"Matriz {tarefa}/{modelo} não soma 80: {arr.sum()}")
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    im = ax.imshow(arr, cmap="Blues", vmin=0)
    ax.set_xticks(range(len(nomes)), nomes, rotation=17, ha="right")
    ax.set_yticks(range(len(nomes)), nomes)
    ax.set_xlabel("Prevista")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusão — {tarefa} / {modelo}")
    for i in range(len(nomes)):
        for j in range(len(nomes)):
            val = arr[i, j]
            ax.text(j, i, str(val), ha="center", va="center", color="white" if val >= .55 * arr.max() else "black")
    fig.colorbar(im, ax=ax, shrink=.8)
    fig_save(f"fig_04_matriz_{tarefa.lower()}_{modelo.lower()}.png")

# ====================== VERIFICAÇÕES ======================
check = []
def add_check(nome, ok, detalhe=""):
    check.append({"verificacao": nome, "ok": bool(ok), "detalhes": detalhe})

for (tarefa, modelo), sub in previsoes.groupby(["tarefa", "modelo"]):
    add_check(f"80 previsões — {tarefa}/{modelo}", len(sub) == 80, f"{len(sub)} linhas")
    add_check(f"URLs únicas — {tarefa}/{modelo}", sub["url"].nunique() == 80, f"{sub['url'].nunique()} URLs")
    registro = final.loc[(final.tarefa == tarefa) & (final.modelo == modelo)]
    if len(registro) != 1:
        add_check(f"Chave de métricas — {tarefa}/{modelo}", False, "Registro ausente ou duplicado")
        continue
    accuracy_recalculada = (sub.real == sub.previsto).mean()
    accuracy_salva = float(registro.accuracy.iloc[0])
    add_check(f"Accuracy consistente — {tarefa}/{modelo}", abs(accuracy_recalculada - accuracy_salva) < 1e-10, f"previsões={accuracy_recalculada:.6f}; CSV={accuracy_salva:.6f}")

for tarefa, modelo in selecoes:
    m = matrizes.loc[(matrizes.tarefa == tarefa) & (matrizes.modelo == modelo) & (matrizes.tipo_avaliacao == "original")]
    add_check(f"Soma matriz — {tarefa}/{modelo}", int(m.quantidade.sum()) == 80, f"{int(m.quantidade.sum())}")
    cs = classes.loc[(classes.tarefa == tarefa) & (classes.modelo == modelo)]
    add_check(f"Suporte classes — {tarefa}/{modelo}", int(cs.support.sum()) == 80, f"{int(cs.support.sum())}")

add_check("Comparação binária: duas abordagens", set(comum.abordagem) == {"XGBoost_multiclasse_agrupado", "XGBoost_binario_direto"} and len(comum) == 2)

# Todos os modelos devem ter previsto exatamente as mesmas URLs.
grupos_url = [set(sub["url"].astype(str)) for _, sub in previsoes.groupby(["tarefa", "modelo"])]
add_check("Mesmo conjunto de 80 URLs para todos os modelos", bool(grupos_url) and all(u == grupos_url[0] for u in grupos_url), "Comparação emparelhada no teste")

# Matrizes precisam ser reconstruíveis a partir das previsões, célula por célula.
for tarefa, modelo in sorted(selecoes):
    sub = previsoes.loc[(previsoes.tarefa == tarefa) & (previsoes.modelo == modelo)]
    nomes = ["Overlap", "Truncation", "Ellipsis"] if tarefa == "Multiclasse" else ["Overlap", "Truncation/Ellipsis"]
    m = matrizes.loc[(matrizes.tarefa == tarefa) & (matrizes.modelo == modelo) & (matrizes.tipo_avaliacao == "original")]
    consistente = len(sub) == 80
    for i, real in enumerate(nomes):
        for j, previsto in enumerate(nomes):
            n_csv = m.loc[(m.classe_real == real) & (m.classe_prevista == previsto), "quantidade"]
            n_pred = int(((sub.real == i) & (sub.previsto == j)).sum())
            consistente = consistente and len(n_csv) == 1 and int(n_csv.iloc[0]) == n_pred
    add_check(f"Matriz reproduz previsões — {tarefa}/{modelo}", consistente)

# A comparação binária comum também pode ser reconstituída das previsões individuais.
from sklearn.metrics import f1_score, accuracy_score
multi_xgb = previsoes.loc[(previsoes.tarefa == "Multiclasse") & (previsoes.modelo == "XGBoost"), ["url", "real", "previsto"]].rename(columns={"real":"real_multi", "previsto":"pred_multi"})
bin_xgb = previsoes.loc[(previsoes.tarefa == "Binario") & (previsoes.modelo == "XGBoost"), ["url", "real", "previsto"]].rename(columns={"real":"real_bin", "previsto":"pred_bin"})
pares = multi_xgb.merge(bin_xgb, on="url", validate="one_to_one")
if len(pares) == 80:
    esperado_bin = np.where(pares.real_multi == 0, 0, 1)
    add_check("Rótulos binários coerentes com multiclasse", bool(np.array_equal(esperado_bin, pares.real_bin.to_numpy())))
    mapa_pred = {"XGBoost_multiclasse_agrupado": np.where(pares.pred_multi == 0, 0, 1), "XGBoost_binario_direto": pares.pred_bin.to_numpy()}
    for nome, pred in mapa_pred.items():
        reg = comum.loc[comum.abordagem == nome]
        ok = len(reg) == 1
        if ok:
            ok = abs(float(reg.f1_macro.iloc[0]) - f1_score(pares.real_bin, pred, average="macro")) < 1e-10
            ok = ok and abs(float(reg.accuracy.iloc[0]) - accuracy_score(pares.real_bin, pred)) < 1e-10
        add_check(f"Comparação comum reprodutível — {nome}", ok)
else:
    add_check("Pareamento da comparação comum", False, f"{len(pares)} pares")

check_df = pd.DataFrame(check)
salvar_csv(check_df, "verificacoes_consistencia.csv")

# ====================== EXPERIMENTOS ANTERIORES ======================
registro_anteriores = []
for chave in ("entradas", "representacoes", "balanceamento", "hiperparametros"):
    antigo = ler(chave)
    if antigo is None:
        continue
    if {"tarefa", "modelo", "f1_macro_mean"}.issubset(antigo.columns):
        resumido = antigo.copy()
        resumido["experimento"] = chave
        cols = ["experimento", "tarefa", "modelo", "f1_macro_mean"]
        for c in ("representacao", "estrategia", "balanceamento", "parametros"):
            if c in resumido.columns:
                cols.append(c)
        resumido[cols].to_csv(OUT / f"arquivo_auxiliar_{chave}.csv", index=False)
        registro_anteriores.append((chave, len(antigo), "OK"))
    else:
        registro_anteriores.append((chave, len(antigo), "Estrutura inesperada: verificar colunas"))
        warnings.append(f"Colunas incompletas na consolidação de {chave}: {list(antigo.columns)}")

# ====================== EXPERIMENTO 1 ======================
baseline = ler("baseline")
if baseline is not None:
    # O script de baseline usa coluna 'problema' no lugar de 'tarefa'.
    base = baseline.rename(columns={"problema": "tarefa"}).copy()
    if {"tarefa", "modelo", "f1_macro"}.issubset(base.columns):
        salvar_csv(base, "tabela_00_baselines.csv")
        registro_anteriores.insert(0, ("baseline", len(base), "OK"))
    else:
        warnings.append("Baseline possui colunas inesperadas; verificar manualmente")

# ====================== RELATÓRIO ======================
def markdown_simples(df, decimais=4):
    """Produz tabela Markdown sem depender de tabulate."""
    valores = df.copy()
    for coluna in valores.select_dtypes(include="number").columns:
        valores[coluna] = valores[coluna].map(lambda v: f"{v:.{decimais}f}")
    valores = valores.fillna("").astype(str)
    cabecalho = "| " + " | ".join(valores.columns) + " |"
    divisor = "| " + " | ".join(["---"] * len(valores.columns)) + " |"
    linhas_md = ["| " + " | ".join(linha) + " |" for linha in valores.itertuples(index=False, name=None)]
    return "\n".join([cabecalho, divisor] + linhas_md)


linhas = [
    "# Consolidação dos experimentos — i18n/l10n",
    "",
    "## Protocolo da avaliação final",
    "- 318 issues de desenvolvimento; 80 issues reservadas para teste.",
    "- Quatro configurações selecionadas previamente por validação cruzada.",
    "- A consolidação lê os CSVs existentes: não treina, não seleciona e não altera modelos.",
    "- A comparação multiclasse vs. binária direta usa métricas de tarefas diferentes; para um confronto controlado em rótulos, usar `tabela_03_comparacao_xgboost_binario.csv`.",
    "- O teste possui somente cinco registros de Ellipsis: relatar contagens absolutas e evitar conclusões fortes.",
    "",
    "## Métricas gerais — teste",
    "",
    markdown_simples(final[["tarefa", "modelo", "accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro"]]),
    "",
    "## Comparação comum (XGBoost)",
    "",
    markdown_simples(comparar[["abordagem", "accuracy", "balanced_accuracy", "f1_macro"]]),
    "",
    "## Verificações",
    f"- {int(check_df.ok.sum())}/{len(check_df)} verificações passaram.",
]
if not check_df.ok.all():
    linhas += ["", "### Verificações que falharam"] + [f"- {r.verificacao}: {r.detalhes}" for r in check_df.loc[~check_df.ok].itertuples(index=False)]
linhas += ["", "## Arquivos anteriores encontrados"]
linhas += [f"- {nome}: {num} linhas; {status}" for nome, num, status in registro_anteriores] or ["- Nenhum resumo experimental opcional encontrado."]
linhas += ["", "## Avisos"] + ([f"- {w}" for w in warnings] if warnings else ["- Nenhum aviso."])
linhas += ["", "## Revisão metodológica posterior", "- Conferir coleta e filtros iniciais; duplicatas de URL e texto; separação estratificada; pipelines e comparabilidade de folds; seleção de configurações; ausência de vazamento; arquivos redundantes/intermediários.", "- Não modificar a seleção dos modelos em função do teste independente."]
(OUT / "relatorio_consolidacao.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")

print("\n=== CONSOLIDAÇÃO CONCLUÍDA ===")
print(final[["tarefa", "modelo", "accuracy", "f1_macro"]].round(4).to_string(index=False))
print(f"\nVerificações: {int(check_df.ok.sum())}/{len(check_df)} OK")
for w in warnings:
    print("AVISO:", w)
print("\nResultados em:", OUT)
if not check_df.ok.all():
    sys.exit("ATENÇÃO: algumas verificações falharam. Consulte verificacoes_consistencia.csv")
