"""
Analise causal do experimento sintetico (spec 0002 secao 3).

Fluxo: gerar populacao -> sortear bracos -> observar o desfecho de cada braco ->
estimar o efeito (Delta de recompra) por estrato e no total, com IC -> traduzir em
margem incremental por tratado com IC CONJUNTO (efeito x valor do cliente) ->
aplicar a regra de decisao pre-registrada.

DADOS SINTETICOS, DEMONSTRACAO DE METODO.
"""
import numpy as np
import pandas as pd

from experimento_causal import config as cfg
from experimento_causal.dgp import CenarioDGP, gerar_populacao
from experimento_causal.randomizacao import sortear

Z = 1.959963985  # normal 97,5%


def observar(df: pd.DataFrame, cen: CenarioDGP, seed: int = cfg.SEED) -> pd.DataFrame:
    """Desfecho realmente observado em cada braco (o resto de Y0/Y1 fica latente)."""
    rng = np.random.default_rng(seed + 1)
    out = df.copy()
    y = np.where(out["braco"].to_numpy() == "tratamento", out["y1"], out["y0"]).astype(int)

    # contaminacao do controle: parte do controle recebe a acao por fora -> ve Y1
    contaminado = (out["braco"].to_numpy() == "controle") & \
                  (out["u_contaminacao"].to_numpy() < cen.contaminacao_controle)
    y = np.where(contaminado, out["y1"], y)
    out["contaminado"] = contaminado

    # ancora de janela na acao: o controle ganha ~3 dias a mais de observacao ->
    # vies para cima na recompra do controle -> ATE puxado para baixo
    if cen.ancora_janela == "acao":
        extra = (out["braco"].to_numpy() == "controle") & (y == 0) & \
                (rng.uniform(0, 1, len(out)) < 0.006)
        y = np.where(extra, 1, y)

    out["y_obs"] = y
    return out


def simular(cen: CenarioDGP) -> pd.DataFrame:
    """Populacao -> sorteio -> desfecho observado, com a mesma semente em toda etapa."""
    return observar(sortear(gerar_populacao(cen), seed=cen.seed), cen, seed=cen.seed)


def _delta_ic(p_t, n_t, p_c, n_c):
    delta = p_t - p_c
    se = np.sqrt(p_t * (1 - p_t) / max(n_t, 1) + p_c * (1 - p_c) / max(n_c, 1))
    return delta, se, delta - Z * se, delta + Z * se


def estimar(df: pd.DataFrame) -> pd.DataFrame:
    """Delta de recompra por estrato efetivo e linha 'total' (media ponderada por n)."""
    linhas = []
    for est, g in df.groupby("estrato_efetivo"):
        t = g.loc[g["braco"] == "tratamento", "y_obs"]
        c = g.loc[g["braco"] == "controle", "y_obs"]
        delta, se, lo, hi = _delta_ic(t.mean(), len(t), c.mean(), len(c))
        linhas.append({"estrato": int(est), "n": len(g),
                       "recompra_tratamento": t.mean(), "recompra_controle": c.mean(),
                       "delta": delta, "se": se, "ic_baixo": lo, "ic_alto": hi,
                       "efeito_verdadeiro": g["tau_verdadeiro"].mean()})
    res = pd.DataFrame(linhas)

    w = res["n"] / res["n"].sum()
    delta_tot = float((w * res["delta"]).sum())
    se_tot = float(np.sqrt((w ** 2 * res["se"] ** 2).sum()))
    res.loc[len(res)] = {"estrato": -1, "n": int(res["n"].sum()),
                         "recompra_tratamento": np.nan, "recompra_controle": np.nan,
                         "delta": delta_tot, "se": se_tot,
                         "ic_baixo": delta_tot - Z * se_tot,
                         "ic_alto": delta_tot + Z * se_tot,
                         "efeito_verdadeiro": float((w * res["efeito_verdadeiro"]).sum())}
    return res


def ic_conjunto(delta, se, n_mc=50_000, seed=cfg.SEED,
                valor_min=cfg.VALOR_CLIENTE_RETIDO_MIN,
                valor_max=cfg.VALOR_CLIENTE_RETIDO_MAX, custo=cfg.CUSTO_ACAO):
    """
    Margem incremental por tratado = Delta * valor_cliente_retido - custo.
    Propaga a incerteza do efeito (Normal) E a do valor do cliente (Uniforme na
    faixa REVALIDAR). Plantar o valor no ponto medio daria um IC falsamente estreito.
    """
    rng = np.random.default_rng(seed + 2)
    d = rng.normal(delta, se, n_mc)
    v = rng.uniform(valor_min, valor_max, n_mc)
    margem = d * v - custo
    return {
        "margem_p2_5": float(np.percentile(margem, 2.5)),
        "margem_mediana": float(np.percentile(margem, 50)),
        "margem_p97_5": float(np.percentile(margem, 97.5)),
        "prob_margem_positiva": float((margem > 0).mean()),
    }


def decidir(delta, se, **kw):
    """Regra pre-registrada (spec secao 3), com IC conjunto efeito x valor do cliente."""
    ic = ic_conjunto(delta, se, **kw)
    if ic["margem_p2_5"] > 0:
        veredito = "escalar"
    elif ic["margem_mediana"] <= 0 and delta - Z * se <= 0:
        veredito = "nao escalar"
    else:
        veredito = "zona morta / depende do valor do cliente"
    return {**ic, "veredito": veredito}


def validar_recuperacao(cen: CenarioDGP | None = None, n_rep: int = 60) -> pd.DataFrame:
    """
    O estimador e nao-viesado? Roda n_rep replicas com seeds diferentes e compara o
    Delta estimado com o efeito plantado, por estrato. Vies proximo de zero e cobertura
    do IC 95% proxima de 0,95 sao a prova de que a maquina de analise esta correta.
    """
    base = cen or CenarioDGP()
    reg = []
    for s in range(n_rep):
        c = CenarioDGP(**{**base.__dict__, "seed": 1000 + s})
        for _, r in estimar(simular(c)).iterrows():
            if r["estrato"] == -1:
                continue
            reg.append({"estrato": int(r["estrato"]), "delta": r["delta"],
                        "verdadeiro": r["efeito_verdadeiro"],
                        "cobre": bool(r["ic_baixo"] <= r["efeito_verdadeiro"] <= r["ic_alto"])})
    d = pd.DataFrame(reg)
    g = d.groupby("estrato").agg(
        delta_medio=("delta", "mean"),
        efeito_verdadeiro=("verdadeiro", "mean"),
        cobertura_ic95=("cobre", "mean"),
        n_replicas=("delta", "size"),
    ).reset_index()
    g.insert(3, "vies", g["delta_medio"] - g["efeito_verdadeiro"])
    return g


def analisar(cen: CenarioDGP | None = None) -> dict:
    cen = cen or CenarioDGP()
    df = simular(cen)
    est = estimar(df)

    por_estrato = []
    for _, r in est.iterrows():
        alvo = "total" if r["estrato"] == -1 else f"estrato {int(r['estrato'])}"
        d = decidir(r["delta"], r["se"])
        por_estrato.append({
            "alvo": alvo, "n": int(r["n"]),
            "delta": round(r["delta"], 4), "delta_verdadeiro": round(r["efeito_verdadeiro"], 4),
            "ic_delta": [round(r["ic_baixo"], 4), round(r["ic_alto"], 4)],
            "margem_mediana": round(d["margem_mediana"], 2),
            "margem_ic95": [round(d["margem_p2_5"], 2), round(d["margem_p97_5"], 2)],
            "prob_margem_positiva": round(d["prob_margem_positiva"], 3),
            "veredito": d["veredito"],
        })

    return {
        "rotulo": cfg.ROTULO_SINTETICO,
        "cenario": df.attrs["cenario"],
        "taxa_contaminacao_controle_realizada": round(
            float(df.loc[df["braco"] == "controle", "contaminado"].mean()), 4),
        "break_even_lift": {"otimista": round(cfg.BREAKEVEN_LIFT_OTIMISTA, 4),
                            "conservador": round(cfg.BREAKEVEN_LIFT_CONSERVADOR, 4)},
        "resultados": por_estrato,
    }


if __name__ == "__main__":
    print(f"Validacao do estimador ({cfg.ROTULO_SINTETICO}, 60 replicas, cenario heterogeneo):")
    print(validar_recuperacao(CenarioDGP(efeito="heterogeneo")).to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))

    for chave in ("heterogeneo", "nulo", "rentavel", "forte"):
        r = analisar(CenarioDGP(efeito=chave))
        print(f"\n=== '{chave}'  ({cfg.ROTULO_SINTETICO}) ===")
        for x in r["resultados"]:
            print(f"  {x['alvo']:<10} n={x['n']:5d}  Delta={x['delta']:+.4f} "
                  f"(verdadeiro {x['delta_verdadeiro']:+.4f})  "
                  f"P(margem>0)={x['prob_margem_positiva']:.2f}  -> {x['veredito']}")
