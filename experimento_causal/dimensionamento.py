"""
Tamanho de amostra, duracao e criterio de decisao (spec 0002 secao 3).

Numeros marcados REVALIDAR (p0 de recompra, valor do cliente retido) entram como
faixa declarada. A saida principal e a analise de sensibilidade: como o n e a
duracao mudam ao longo da faixa de p0, e o que o teto de duracao (6 meses) implica.
"""
import numpy as np
import pandas as pd
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

from experimento_causal import config as cfg

# razao tratamento/controle vem direto da fracao de controle do desenho
RAZAO_TRAT_CTRL = (1 - cfg.FRACAO_CONTROLE) / cfg.FRACAO_CONTROLE


def elegiveis_mes(recomputar=False):
    """
    Volume mensal de elegiveis por estrato. Valores fixos abaixo (evita carregar o
    scorer de 6 MB no import); `recomputar=True` regenera de
    calibracao_modelo.resumo_elegibilidade() se o scorer mudar.
    """
    if recomputar:
        from experimento_causal.calibracao_modelo import resumo_elegibilidade
        e = resumo_elegibilidade()
        return {"total": e["elegivel_por_mes"],
                **{i: s["n"] for i, s in enumerate(e["estratos"])}}
    return {"total": 1806, 0: 305, 1: 458, 2: 1043}  # scorer atual; reconferir com recomputar=True


ELEGIVEIS_MES = elegiveis_mes()


def n_por_braco(p0, mde, alfa=cfg.ALFA, poder=cfg.PODER, razao=RAZAO_TRAT_CTRL):
    """
    n de controle e de tratamento para detectar um lift 'mde' sobre uma base p0,
    num teste de duas proporcoes com alocacao desigual (padrao 75/25).
    """
    h = proportion_effectsize(min(p0 + mde, 0.999), p0)
    n_ctrl = NormalIndPower().solve_power(
        effect_size=h, alpha=alfa, power=poder, ratio=razao, alternative="two-sided")
    n_ctrl = int(np.ceil(n_ctrl))
    n_trat = int(np.ceil(n_ctrl * razao))
    return n_ctrl, n_trat


def duracao_meses(n_total, elegiveis_mes):
    return n_total / elegiveis_mes


def sensibilidade_p0(mde, p0_grid=None, escopo="total"):
    """n e duracao ao longo da faixa de p0 REVALIDAR, para um lift alvo fixo."""
    p0_grid = p0_grid if p0_grid is not None else np.round(
        np.arange(cfg.P0_RECOMPRA_MIN, cfg.P0_RECOMPRA_MAX + 1e-9, 0.02), 2)
    vol = ELEGIVEIS_MES[escopo]
    linhas = []
    for p0 in p0_grid:
        nc, nt = n_por_braco(p0, mde)
        n_total = nc + nt
        linhas.append({"p0": float(p0), "mde": mde, "n_controle": nc, "n_tratamento": nt,
                       "n_total": n_total, "duracao_meses": round(duracao_meses(n_total, vol), 2),
                       "cabe_no_teto": duracao_meses(n_total, vol) <= cfg.DURACAO_MAX_MESES})
    return pd.DataFrame(linhas)


def tabela_mde(p0=0.10):
    """Para um p0 fixo, o n e a duracao sob os dois break-evens e um alvo intermediario."""
    alvos = {
        "break-even otimista (LTV R$350)": cfg.BREAKEVEN_LIFT_OTIMISTA,
        "intermediario (+0,05)": 0.05,
        "break-even conservador (margem R$105)": cfg.BREAKEVEN_LIFT_CONSERVADOR,
    }
    linhas = []
    for nome, mde in alvos.items():
        nc, nt = n_por_braco(p0, mde)
        for escopo in ("total", 0):
            vol = ELEGIVEIS_MES[escopo]
            linhas.append({
                "alvo": nome, "mde": round(mde, 3), "escopo": str(escopo),
                "n_total": nc + nt,
                "duracao_meses": round(duracao_meses(nc + nt, vol), 2),
                "cabe_no_teto_6m": duracao_meses(nc + nt, vol) <= cfg.DURACAO_MAX_MESES,
            })
    return pd.DataFrame(linhas)


def criterio_pre_registrado():
    return {
        "desfecho_primario": "recompra em 90 dias, em margem incremental por cliente tratado",
        "regra_de_escala": ("escalar so se o limite inferior do IC 95% CONJUNTO "
                            "(efeito x valor_cliente_retido) da margem por tratado for > 0"),
        "zona_morta": "IC conjunto contem zero mas mediana positiva -> nao escala, reformula",
        "teto_duracao_meses": cfg.DURACAO_MAX_MESES,
        "veredito_no_teto": ("se ao fim de 6 meses o IC ainda cruza o break-even: "
                             "efeito pequeno demais para pagar no volume atual -> nao escala"),
        "guardrails": ["taxa de detrator entre respondentes nao piora no tratado",
                       "reclamacoes e contatos de SAC nao sobem no tratado"],
        "revalidar_antes_de_executar": ["p0 de recompra 90d sem acao (piloto ou CRM)",
                                        "valor_cliente_retido (margem de contribuicao, nao LTV de brochura)"],
    }


if __name__ == "__main__":
    print(cfg.ROTULO_SINTETICO)
    print("\nn e duracao por alvo de lift (p0 = 0,10):")
    print(tabela_mde().to_string(index=False))
    print("\nSensibilidade a p0 (alvo de lift = break-even otimista "
          f"{cfg.BREAKEVEN_LIFT_OTIMISTA:.3f}, escopo estrato 0):")
    print(sensibilidade_p0(cfg.BREAKEVEN_LIFT_OTIMISTA, escopo=0).to_string(index=False))
