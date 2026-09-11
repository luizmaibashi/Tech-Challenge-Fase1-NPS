"""
Guarda de sincronizacao entre os dois modelos treinados no projeto (ADR-0003).

`models/v1/pipeline_completo.pkl` (producao, multiclasse, ADR-0001) e
`models/v1/risco_detrator.pkl` (scorer binario dedicado do experimento causal,
ADR-0002/0015) sao treinados separadamente de proposito — calibrar o primeiro
diretamente degradava o ECE de 0,0125 para 0,0478 por falta de dado de calibracao
disjunto (ver ADR-0003). O risco aceito por manter os dois e a divergencia silenciosa
se um for retreinado sem o outro; este teste e o que pega isso.

NAO compara os dois por limiar absoluto: p_producao (multiclasse, 3 classes
competindo pela massa de probabilidade) e p_experimento (binario, 2 classes) tem
escalas estruturalmente diferentes por design — na base atual, mediana 0,611 vs
0,873. Comparar contra o mesmo corte (0,60) e invalido, nao evidencia de
divergencia (medido nesta investigacao: 21,5% "divergiam" so por causa da escala,
com os dois modelos ainda concordando sobre quem e mais arriscado). O que importa
para o proposito deste ADR e se o RANKING relativo entre os dois se mantem —
Spearman mede exatamente isso, independente de nivel absoluto.

Fail-closed: se a correlacao cair abaixo do piso, o teste fica vermelho — nao
passa por omissao.
"""
import joblib
import pandas as pd
from scipy.stats import spearmanr

from experimento_causal import config as cfg
from experimento_causal.calibracao_modelo import prever_risco
from utils import criar_features, FEATURES_MODELO

# piso medido na investigacao do ADR-0003 (0,9767, CV externa, dado real da
# Fase 1); folga deliberada abaixo do valor medido para nao ficar fragil a
# reamostragem/seed em retreinos futuros.
CORRELACAO_MINIMA = 0.95


def _scores():
    df = pd.read_csv(cfg.DATA_PATH).drop(columns=cfg.LEAKAGE_COLS)
    df = criar_features(df)

    pipeline = joblib.load(cfg.MODELO_PATH)
    p_producao = pipeline.predict_proba(df[FEATURES_MODELO])[:, 0]  # classe 0 = Detrator

    p_experimento = prever_risco(df)  # scorer binario dedicado, ja recalibrado

    return p_producao, p_experimento


def test_scores_dos_dois_modelos_sao_fortemente_correlacionados():
    p_producao, p_experimento = _scores()
    rho, _ = spearmanr(p_producao, p_experimento)
    assert rho >= CORRELACAO_MINIMA, (
        f"correlacao caiu para {rho:.4f} (piso {CORRELACAO_MINIMA}) — "
        "pipeline_completo.pkl e risco_detrator.pkl divergiram no ranking de "
        "risco; retreinar o scorer do experimento "
        "(calibracao_modelo.treinar_scorer) apos qualquer mudanca em "
        "train_pipeline.py (ADR-0003)"
    )
