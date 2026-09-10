"""
Gera as figuras e os artefatos versionados do arco causal (ticket 0015).

Saidas em reports/experimento_causal/:
  reliability_v1.png          (produzido por calibracao_modelo)
  efeito_por_estrato.png       Delta estimado vs plantado, por estrato, com IC
  validacao_estimador.png      vies e cobertura do IC 95% em 80 replicas
  decisao_superficie.png       margem incremental por tratado vs valor do cliente
  sensibilidade_p0.png         n e duracao vs p0, com o teto de 6 meses
  pavc_cenarios.md             veredito com e sem cada mitigacao do PAVC
  resultados.json              numeros de todos os cenarios

Tudo rotulado "dados sinteticos, demonstracao de metodo".
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experimento_causal import config as cfg
from experimento_causal.analise import (analisar, estimar, observar,
                                        validar_recuperacao, Z)
from experimento_causal.dgp import CenarioDGP, gerar_populacao
from experimento_causal.dimensionamento import sensibilidade_p0, ELEGIVEIS_MES
from experimento_causal.randomizacao import sortear

C_TRAT, C_CTRL, C_VERD = "#1b7837", "#762a83", "#d95f02"
_ROT = dict(fontsize=8, color="#666")


def _rotulo(ax):
    ax.text(0.99, 0.01, cfg.ROTULO_SINTETICO, transform=ax.transAxes,
            ha="right", va="bottom", **_ROT)


def fig_efeito_por_estrato(cen=None):
    cen = cen or CenarioDGP(efeito="heterogeneo")
    df = observar(sortear(gerar_populacao(cen), seed=cen.seed), cen, seed=cen.seed)
    est = estimar(df)
    est = est[est["estrato"] >= 0]
    x = np.arange(len(est))

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.errorbar(x - 0.09, est["delta"], yerr=Z * est["se"], fmt="o", color=C_TRAT,
                capsize=4, label="Delta estimado (IC 95%)")
    ax.plot(x + 0.09, est["efeito_verdadeiro"], "s", color=C_VERD,
            label="efeito verdadeiro plantado")
    ax.axhline(cfg.BREAKEVEN_LIFT_OTIMISTA, ls="--", color="#888",
               label=f"break-even otimista {cfg.BREAKEVEN_LIFT_OTIMISTA:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"estrato {i}\n{f}" for i, f in
                        zip(est['estrato'], [f'[{a},{b})' for a, b in cfg.FAIXAS_P])])
    ax.set_ylabel("lift na recompra em 90 dias")
    ax.set_title("Uma replica do experimento: o IC 95% cobre o efeito verdadeiro\n"
                 "em cada estrato (cenario heterogeneo). O nao-vies vem da validacao "
                 "em 80 replicas.")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    _rotulo(ax)
    fig.tight_layout()
    fig.savefig(cfg.REPORTS_DIR / "efeito_por_estrato.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_validacao_estimador(n_rep=80):
    v = validar_recuperacao(CenarioDGP(efeito="heterogeneo"), n_rep=n_rep)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(v))
    a1.bar(x, v["vies"], color=C_TRAT)
    a1.axhline(0, color="#333", lw=0.8)
    a1.set_xticks(x); a1.set_xticklabels([f"estrato {i}" for i in v["estrato"]])
    a1.set_ylabel("vies (Delta medio - efeito verdadeiro)")
    a1.set_title(f"Vies do estimador ({n_rep} replicas)")
    a1.set_ylim(-0.02, 0.02)
    a1.grid(alpha=0.3, axis="y")

    a2.bar(x, v["cobertura_ic95"], color=C_CTRL)
    a2.axhline(0.95, color="#d95f02", ls="--", label="nominal 0,95")
    a2.set_xticks(x); a2.set_xticklabels([f"estrato {i}" for i in v["estrato"]])
    a2.set_ylabel("cobertura do IC 95%")
    a2.set_ylim(0.8, 1.0)
    a2.set_title("Cobertura do IC 95%")
    a2.legend(fontsize=8); a2.grid(alpha=0.3, axis="y")
    _rotulo(a2)
    fig.suptitle("Prova de que a maquina de analise esta correta", fontsize=12)
    fig.tight_layout()
    fig.savefig(cfg.REPORTS_DIR / "validacao_estimador.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return v


def fig_decisao_superficie(cen=None):
    cen = cen or CenarioDGP(efeito="rentavel")
    df = observar(sortear(gerar_populacao(cen), seed=cen.seed), cen, seed=cen.seed)
    est = estimar(df)
    valores = np.linspace(80, 380, 120)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    cores = ["#2166ac", "#4393c3", "#92c5de", "#333333"]
    for (_, r), cor in zip(est.iterrows(), cores):
        alvo = "total" if r["estrato"] == -1 else f"estrato {int(r['estrato'])}"
        margem = r["delta"] * valores - cfg.CUSTO_ACAO
        lo = (r["delta"] - Z * r["se"]) * valores - cfg.CUSTO_ACAO
        ax.plot(valores, margem, color=cor, label=alvo,
                lw=2 if r["estrato"] == -1 else 1.4,
                ls="--" if r["estrato"] == -1 else "-")
        ax.fill_between(valores, lo, margem, color=cor, alpha=0.12)
    ax.axhline(0, color="#333", lw=1)
    ax.axvspan(cfg.VALOR_CLIENTE_RETIDO_MIN, cfg.VALOR_CLIENTE_RETIDO_MAX,
               color="#fdae61", alpha=0.15, label="faixa REVALIDAR do valor do cliente")
    ax.set_xlabel("valor de um cliente retido (R$)")
    ax.set_ylabel("margem incremental por cliente tratado (R$)")
    ax.set_title("A decisao de escalar depende do valor do cliente retido\n"
                 f"(cenario rentavel; margem = Delta x valor - R$ {cfg.CUSTO_ACAO:.0f})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _rotulo(ax)
    fig.tight_layout()
    fig.savefig(cfg.REPORTS_DIR / "decisao_superficie.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_sensibilidade_p0():
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for escopo, cor in [("total", C_TRAT), (0, C_CTRL)]:
        s = sensibilidade_p0(cfg.BREAKEVEN_LIFT_OTIMISTA, escopo=escopo)
        rot = "toda a populacao elegivel" if escopo == "total" else "estrato 0 (o menor)"
        ax.plot(s["p0"], s["duracao_meses"], "o-", color=cor, label=rot)
    ax.axhline(cfg.DURACAO_MAX_MESES, color="#d95f02", ls="--",
               label=f"teto pre-registrado ({cfg.DURACAO_MAX_MESES} meses)")
    ax.set_xlabel("p0 REVALIDAR - taxa de recompra 90d sem acao")
    ax.set_ylabel("duracao do experimento (meses)")
    ax.set_title("Duracao para detectar um lift no break-even otimista (0,086)\n"
                 "vs a taxa-base de recompra, que ainda nao foi medida")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _rotulo(ax)
    fig.tight_layout()
    fig.savefig(cfg.REPORTS_DIR / "sensibilidade_p0.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def tabela_pavc():
    """Veredito do total com e sem cada mitigacao do PAVC."""
    linhas = []
    base = CenarioDGP(efeito="forte")
    cenarios = {
        "com todas as mitigacoes (desenho do spec)": base,
        "contaminacao do controle 30% ignorada": CenarioDGP(efeito="forte",
                                                            contaminacao_controle=0.30),
        "janela ancorada na acao, nao na entrega": CenarioDGP(efeito="forte",
                                                             ancora_janela="acao"),
    }
    for nome, cen in cenarios.items():
        r = analisar(cen)
        tot = [x for x in r["resultados"] if x["alvo"] == "total"][0]
        linhas.append({"cenario": nome, "delta_total": tot["delta"],
                       "delta_verdadeiro": tot["delta_verdadeiro"],
                       "prob_margem_positiva": tot["prob_margem_positiva"],
                       "veredito": tot["veredito"]})
    df = pd.DataFrame(linhas)
    txt = ["# Cenarios PAVC - efeito no veredito\n",
           f"> {cfg.ROTULO_SINTETICO}\n",
           "Cenario base: efeito verdadeiro 'forte'. Cada linha mostra o que acontece "
           "quando uma mitigacao do PAVC NAO e aplicada.\n",
           df.to_markdown(index=False, floatfmt=".4f"), ""]
    (cfg.REPORTS_DIR / "pavc_cenarios.md").write_text("\n".join(txt), encoding="utf-8")
    return df


def resultados_json():
    saida = {"rotulo": cfg.ROTULO_SINTETICO,
             "break_even_lift": {"otimista": round(cfg.BREAKEVEN_LIFT_OTIMISTA, 4),
                                 "conservador": round(cfg.BREAKEVEN_LIFT_CONSERVADOR, 4)},
             "elegiveis_por_mes": ELEGIVEIS_MES,
             "cenarios": {}}
    for chave in ("nulo", "heterogeneo", "rentavel", "forte"):
        saida["cenarios"][chave] = analisar(CenarioDGP(efeito=chave))
    with open(cfg.REPORTS_DIR / "resultados.json", "w", encoding="utf-8") as f:
        json.dump(saida, f, indent=2, ensure_ascii=False)
    return saida


def gerar_tudo():
    cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    from experimento_causal.calibracao_modelo import diagnosticar
    diagnosticar(salvar=True, verbose=False)
    fig_efeito_por_estrato()
    fig_validacao_estimador()
    fig_decisao_superficie()
    fig_sensibilidade_p0()
    tabela_pavc()
    resultados_json()
    print(f"figuras e artefatos em {cfg.REPORTS_DIR.relative_to(cfg.RAIZ)}/")


if __name__ == "__main__":
    gerar_tudo()
