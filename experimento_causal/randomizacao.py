"""
Sorteio estratificado tratamento/controle (spec 0002 secao 2).

- unidade = cliente, fixado no braco na primeira entrada (no sintetico cada linha ja
  e uma entrada unica; a fixacao por cliente repetido e abstraida);
- sorteio dentro de cada estrato efetivo, ~75/25;
- colapso de estrato: balde com contagem esperada por braco abaixo do piso
  (config.PISO_CELULA_POR_BRACO) e fundido com o vizinho de risco mais proximo,
  por uma regra fixa, nunca decidida com o dado na mao (PAVC edge case 1).
"""
import numpy as np
import pandas as pd

from experimento_causal import config as cfg


def estrato_efetivo(df: pd.DataFrame, fracao_controle: float = cfg.FRACAO_CONTROLE,
                    piso: int = cfg.PISO_CELULA_POR_BRACO) -> pd.Series:
    """Mapeia cada estrato para o estrato apos colapso de baldes ralos."""
    contagem = df["estrato"].value_counts().sort_index()
    n_min_braco = contagem * min(fracao_controle, 1 - fracao_controle)

    ordem = list(contagem.index)               # estratos ordenados por risco crescente
    mapa = {e: e for e in ordem}
    # funde da esquerda para a direita: um balde ralo vai para o vizinho seguinte
    for i, e in enumerate(ordem[:-1]):
        atual = [k for k, v in mapa.items() if v == mapa[e]]
        if sum(n_min_braco[k] for k in atual) < piso:
            for k in atual:
                mapa[k] = mapa[ordem[i + 1]]
    # se o ultimo balde ainda ficou ralo, funde com o anterior
    ultimo = mapa[ordem[-1]]
    grupo_ultimo = [k for k, v in mapa.items() if v == ultimo]
    if sum(n_min_braco[k] for k in grupo_ultimo) < piso and len(set(mapa.values())) > 1:
        alvo = sorted(v for v in set(mapa.values()) if v != ultimo)[-1]
        for k in grupo_ultimo:
            mapa[k] = alvo

    return df["estrato"].map(mapa).rename("estrato_efetivo")


def sortear(df: pd.DataFrame, fracao_controle: float = cfg.FRACAO_CONTROLE,
            seed: int = cfg.SEED) -> pd.DataFrame:
    out = df.copy()
    out["estrato_efetivo"] = estrato_efetivo(out, fracao_controle)
    rng = np.random.default_rng(seed)

    # os chamadores passam um RangeIndex (gerar_populacao / reset_index), entao
    # posicao == rotulo e o fancy-index posicional abaixo e valido.
    braco = np.empty(len(out), dtype=object)
    for pos in out.groupby("estrato_efetivo").indices.values():
        pos = pos.copy()
        rng.shuffle(pos)
        n_ctrl = int(round(len(pos) * fracao_controle))
        braco[pos] = "tratamento"
        braco[pos[:n_ctrl]] = "controle"
    out["braco"] = braco
    return out


def checar_balanco(df: pd.DataFrame, covariaveis=("risco_calibrado", "detrator",
                                                  "delivery_delay_days",
                                                  "customer_tenure_months")) -> pd.DataFrame:
    """Diferenca padronizada de media entre bracos. |d| < 0,1 e o alvo usual."""
    linhas = []
    for c in covariaveis:
        if c not in df.columns:
            continue
        t = df.loc[df["braco"] == "tratamento", c].astype(float)
        k = df.loc[df["braco"] == "controle", c].astype(float)
        dp = np.sqrt((t.var() + k.var()) / 2) or 1.0
        linhas.append({"covariavel": c, "media_tratamento": t.mean(),
                       "media_controle": k.mean(),
                       "dif_padronizada": (t.mean() - k.mean()) / dp})
    return pd.DataFrame(linhas)


if __name__ == "__main__":
    from experimento_causal.dgp import gerar_populacao, CenarioDGP

    pop = sortear(gerar_populacao(CenarioDGP(efeito="heterogeneo")))
    print(cfg.ROTULO_SINTETICO)
    print("\nEstratos originais -> efetivos:")
    print(pop.groupby(["estrato", "estrato_efetivo"]).size().rename("n"))
    print("\nBracos por estrato efetivo:")
    print(pd.crosstab(pop["estrato_efetivo"], pop["braco"]))
    print("\nBalanco de covariaveis:")
    print(checar_balanco(pop).to_string(index=False,
          float_format=lambda v: f"{v:.4f}"))
