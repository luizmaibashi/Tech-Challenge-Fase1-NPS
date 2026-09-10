"""
Gerador de dados sinteticos do experimento (ticket 0015).

DADOS SINTETICOS, DEMONSTRACAO DE METODO. O efeito verdadeiro do cupom e PLANTADO
aqui (EFEITO_POR_ESTRATO, no topo deste modulo). O prototipo serve para mostrar que a
analise recupera o que foi plantado e aplica a regra de decisao economica; nao afirma
nada sobre o cupom real.

Como funciona:
- a populacao elegivel sintetica reamostra clientes reais da Fase 1 que passam no
  gate (risco recalibrado >= 0,60), preservando estrato e rotulo de detrator reais
  (as densidades por estrato batem com a base);
- para cada cliente sao sorteados os desfechos potenciais Y(0) e Y(1) de recompra
  em 90 dias, com um uniforme latente compartilhado que garante Y(1) >= Y(0) (o
  cupom nunca piora);
- o efeito plantado e heterogeneo por estrato e pode decair por coorte (novidade);
- os cenarios PAVC (contaminacao do controle, ancora de janela na acao) sao toggles.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from utils import criar_features, FEATURES_MODELO
from experimento_causal import config as cfg
from experimento_causal.calibracao_modelo import prever_risco

# --- VERDADE PLANTADA -------------------------------------------------------
# Fica so aqui. Nada fora deste modulo deve importar estes valores: a analise
# tem que recuperar o efeito as cegas. Ancorado na faixa de p0 do config.
MESES_PADRAO = 6
P0_POR_ESTRATO = [0.14, 0.10, 0.06]      # recompra 90d SEM acao; detrator quase certo recompra menos
P0_NAO_DETRATOR = 0.30                   # quem nao ia virar detrator recompra mais
EFEITO_NAO_DETRATOR = 0.01              # cupom quase nao move quem ja estava ok
# lift aditivo na prob de recompra por estrato, so em detrator. Estrato 0 do
# 'heterogeneo' fica acima do break-even otimista (0,086) e abaixo do
# conservador (0,286): a decisao de escalar depende do valor do cliente (ADR-0002).
EFEITO_POR_ESTRATO = {
    "heterogeneo": [0.12, 0.05, 0.012],
    "nulo": [0.0, 0.0, 0.0],
    "rentavel": [0.20, 0.14, 0.07],
    "forte": [0.38, 0.22, 0.09],
}


@dataclass
class CenarioDGP:
    efeito: str = "heterogeneo"           # chave de EFEITO_POR_ESTRATO
    meses: int = MESES_PADRAO
    elegiveis_por_mes: int | None = None  # None = usa o volume real observado
    decaimento_novidade: float = 0.0      # fracao do efeito perdida por mes de coorte
    contaminacao_controle: float = 0.0    # fracao do controle que recebe acao por fora
    ancora_janela: str = "entrega"        # "entrega" (correto) ou "acao" (assimetrico)
    seed: int = cfg.SEED


@lru_cache(maxsize=1)
def _pool_elegivel_real():
    """
    Clientes reais da Fase 1 que passam no gate, com estrato e rotulo de detrator.
    Deterministico (nao depende de seed nem cenario) -> cacheado por processo.
    """
    df = pd.read_csv(cfg.DATA_PATH).drop(columns=cfg.LEAKAGE_COLS)
    df = criar_features(df)
    risco = prever_risco(df)
    detrator = (df["nps_score"] <= cfg.DETRATOR_CUTOFF).to_numpy()

    elegivel = risco >= cfg.P_DETRATOR_ELEGIVEL
    faixas = cfg.FAIXAS_P
    estrato = np.full(len(df), -1)
    for i, (lo, hi) in enumerate(faixas):
        estrato[(risco >= lo) & (risco < hi)] = i

    pool = pd.DataFrame({
        "risco_calibrado": risco,
        "estrato": estrato,
        "detrator": detrator,
        **{c: df[c].to_numpy() for c in FEATURES_MODELO},
    })
    return pool[elegivel & (pool["estrato"] >= 0)].reset_index(drop=True)


def gerar_populacao(cenario: CenarioDGP | None = None) -> pd.DataFrame:
    cen = cenario or CenarioDGP()
    rng = np.random.default_rng(cen.seed)
    pool = _pool_elegivel_real()

    por_mes = cen.elegiveis_por_mes or len(pool)
    efeitos = EFEITO_POR_ESTRATO[cen.efeito]

    blocos = []
    for mes in range(1, cen.meses + 1):
        amostra = pool.sample(n=por_mes, replace=True, random_state=int(rng.integers(1e9)))
        amostra = amostra.reset_index(drop=True)
        amostra["mes_coorte"] = mes
        blocos.append(amostra)
    df = pd.concat(blocos, ignore_index=True)
    df.insert(0, "cliente_id", np.arange(len(df)))

    est = df["estrato"].to_numpy()
    det = df["detrator"].to_numpy()

    # taxa-base de recompra em 90 dias, sem acao
    p0_base = np.where(det, np.array(P0_POR_ESTRATO)[est], P0_NAO_DETRATOR)
    p0 = np.clip(p0_base + rng.normal(0, 0.01, len(df)), 0.01, 0.99)

    # efeito verdadeiro, com decaimento de novidade por coorte. Cenario sem efeito
    # nenhum (todos os estratos zerados) zera tambem o efeito no nao-detrator.
    efeito_nd = EFEITO_NAO_DETRATOR if any(efeitos) else 0.0
    tau_base = np.where(det, np.array(efeitos)[est], efeito_nd)
    fator_novidade = np.clip(1 - cen.decaimento_novidade * (df["mes_coorte"].to_numpy() - 1), 0, 1)
    tau = tau_base * fator_novidade
    p1 = np.clip(p0 + tau, 0.01, 0.99)

    # uniforme latente compartilhado -> monotonicidade individual (cupom nunca piora)
    u = rng.uniform(0, 1, len(df))
    df["y0"] = (u < p0).astype(int)
    df["y1"] = (u < p1).astype(int)
    df["p0_verdadeiro"] = p0
    df["tau_verdadeiro"] = tau

    # marcador para o cenario de contaminacao do controle (realizado na analise)
    df["u_contaminacao"] = rng.uniform(0, 1, len(df))

    df.attrs["cenario"] = {
        "rotulo": cfg.ROTULO_SINTETICO,
        "efeito": cen.efeito,
        "efeito_por_estrato_plantado": efeitos,
        "meses": cen.meses,
        "elegiveis_por_mes": int(por_mes),
        "decaimento_novidade": cen.decaimento_novidade,
        "contaminacao_controle": cen.contaminacao_controle,
        "ancora_janela": cen.ancora_janela,
        "ate_medio_plantado": float(np.mean(df["tau_verdadeiro"])),
    }
    return df


def resumo(df: pd.DataFrame) -> pd.DataFrame:
    """Efeito verdadeiro medio por estrato (o que a analise tem que recuperar)."""
    g = df.groupby("estrato").agg(
        n=("cliente_id", "size"),
        densidade_detrator=("detrator", "mean"),
        p0_medio=("p0_verdadeiro", "mean"),
        efeito_verdadeiro=("tau_verdadeiro", "mean"),
        recompra_y0=("y0", "mean"),
        recompra_y1=("y1", "mean"),
    )
    g["faixa"] = [f"[{lo}, {hi})" for lo, hi in cfg.FAIXAS_P]
    return g.reset_index()


if __name__ == "__main__":
    for chave in ("heterogeneo", "nulo", "rentavel"):
        pop = gerar_populacao(CenarioDGP(efeito=chave))
        print(f"\n=== cenario '{chave}'  ({cfg.ROTULO_SINTETICO}) ===")
        print(f"n total = {len(pop)}   ATE medio plantado = "
              f"{pop.attrs['cenario']['ate_medio_plantado']:.4f}")
        print(resumo(pop).to_string(index=False,
              float_format=lambda v: f"{v:.4f}"))
