"""
Testes do arco de experimentacao causal (spec 0002 secao 5).

Cobrem a logica que decide o resultado: monotonicidade dos desfechos potenciais,
regra de colapso de estrato, nao-vies do estimador e as fronteiras da regra de decisao.
Nao testam os scripts de diagnostico (calibracao) — mesma convencao dos outros
analysis scripts do repo, que tambem nao tem teste de unidade.
"""
import numpy as np
import pytest

from experimento_causal import config as cfg
from experimento_causal.dgp import CenarioDGP, gerar_populacao, resumo
from experimento_causal.randomizacao import estrato_efetivo, sortear
from experimento_causal.analise import analisar, decidir, estimar, observar, validar_recuperacao


# --------------------------------------------------------------------------- DGP
def test_desfecho_potencial_monotono():
    """O cupom nunca piora: y1 >= y0 para todo cliente, em todo cenario."""
    for chave in ("nulo", "heterogeneo", "forte"):
        df = gerar_populacao(CenarioDGP(efeito=chave, meses=2))
        assert (df["y1"] >= df["y0"]).all()


def test_cenario_nulo_tem_efeito_zero():
    df = gerar_populacao(CenarioDGP(efeito="nulo", meses=6))
    assert abs(df["tau_verdadeiro"].mean()) < 1e-9


def test_densidade_de_estrato_bate_com_a_base_real():
    df = gerar_populacao(CenarioDGP(meses=6))
    r = resumo(df).set_index("estrato")
    # densidades observadas na base real (spec secao 2): ~0,72 / 0,92 / 0,998
    assert r.loc[0, "densidade_detrator"] == pytest.approx(0.72, abs=0.06)
    assert r.loc[2, "densidade_detrator"] > 0.97


def test_decaimento_de_novidade_reduz_efeito_das_coortes_tardias():
    df = gerar_populacao(CenarioDGP(efeito="heterogeneo", meses=6, decaimento_novidade=0.15))
    por_mes = df.groupby("mes_coorte")["tau_verdadeiro"].mean()
    assert por_mes.iloc[-1] < por_mes.iloc[0]


# ------------------------------------------------------------------- randomizacao
def test_colapso_de_estrato_funde_balde_ralo():
    df = gerar_populacao(CenarioDGP(meses=6))
    # deixa o estrato 0 com poucas linhas -> deve fundir com o vizinho
    poucos = df[df["estrato"] == 0].head(20)
    resto = df[df["estrato"] != 0]
    combinado = np.r_[poucos.index.values, resto.index.values]
    sub = df.loc[combinado].reset_index(drop=True)
    ef = estrato_efetivo(sub)
    assert ef.nunique() < sub["estrato"].nunique()
    assert 0 not in ef.unique()  # o balde ralo foi absorvido


def test_sorteio_estratificado_respeita_a_fracao_de_controle():
    df = sortear(gerar_populacao(CenarioDGP(meses=6)), fracao_controle=0.25)
    for _, g in df.groupby("estrato_efetivo"):
        frac = (g["braco"] == "controle").mean()
        assert frac == pytest.approx(0.25, abs=0.03)


def test_bracos_ficam_balanceados_nas_covariaveis():
    from experimento_causal.randomizacao import checar_balanco
    b = checar_balanco(sortear(gerar_populacao(CenarioDGP(meses=6))))
    assert (b["dif_padronizada"].abs() < 0.1).all()


# ------------------------------------------------------------------------ analise
def test_estimador_recupera_o_efeito_plantado_sem_vies():
    v = validar_recuperacao(CenarioDGP(efeito="heterogeneo"), n_rep=40)
    assert (v["vies"].abs() < 0.01).all()
    assert (v["cobertura_ic95"].between(0.85, 1.0)).all()


def test_contaminacao_do_controle_puxa_o_efeito_para_baixo():
    limpo = analisar(CenarioDGP(efeito="forte"))
    sujo = analisar(CenarioDGP(efeito="forte", contaminacao_controle=0.30))
    d_limpo = [r for r in limpo["resultados"] if r["alvo"] == "total"][0]["delta"]
    d_sujo = [r for r in sujo["resultados"] if r["alvo"] == "total"][0]["delta"]
    assert d_sujo < d_limpo


def test_ancora_na_acao_enviesa_contra_o_tratamento():
    reto = analisar(CenarioDGP(efeito="forte", ancora_janela="entrega"))
    torto = analisar(CenarioDGP(efeito="forte", ancora_janela="acao"))
    d_reto = [r for r in reto["resultados"] if r["alvo"] == "total"][0]["delta"]
    d_torto = [r for r in torto["resultados"] if r["alvo"] == "total"][0]["delta"]
    assert d_torto <= d_reto


def test_regra_de_decisao_nas_fronteiras():
    # efeito nulo com IC apertado -> nao escalar
    assert decidir(0.0, 0.002)["veredito"] == "nao escalar"
    # efeito grande e IC conjunto todo positivo -> escalar
    assert decidir(0.30, 0.01, valor_min=300, valor_max=350)["veredito"] == "escalar"
    # efeito positivo mas dependente do valor do cliente -> zona morta
    assert "zona morta" in decidir(0.12, 0.01)["veredito"]


def test_saida_sempre_rotulada_como_sintetica():
    r = analisar(CenarioDGP(efeito="nulo"))
    assert r["rotulo"] == cfg.ROTULO_SINTETICO
    assert r["cenario"]["rotulo"] == cfg.ROTULO_SINTETICO
