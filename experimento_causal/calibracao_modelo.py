"""
Pre-requisito do experimento (spec 0002 secao 0): as probabilidades do modelo v1
sao confiaveis como probabilidades?

O corte de elegibilidade P(Detrator) >= 0,35 e os estratos por faixa de P so fazem
sentido se "0,35" corresponder de fato a ~35 por cento de chance de o cliente virar
detrator. RandomForest com class_weight='balanced' costuma empurrar as probabilidades
para longe de 0 e 1, entao isso precisa ser medido, nao presumido.

Este modulo gera as probabilidades OOF (out-of-fold, mesma receita de
threshold_calibration.py) e mede a calibracao com tres numeros:
- Brier score: erro quadratico medio da probabilidade (quanto menor, melhor).
- ECE (Expected Calibration Error): distancia media entre confianca e acerto por bin.
- MCE (Maximum Calibration Error): o pior bin.

Se houver descalibracao relevante, mede quanto uma recalibracao isotonica corrigiria,
avaliada num split separado para nao dar credito otimista.

DADOS SINTETICOS NAO, aqui: roda sobre o dataset real da Fase 1. O rotulo de sintetico
vale para dgp.py em diante.
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from utils import criar_features, FEATURES_MODELO
from experimento_causal import config as cfg


def _carregar_xy():
    df = pd.read_csv(cfg.DATA_PATH).drop(columns=cfg.LEAKAGE_COLS)
    df = criar_features(df)
    X = df[FEATURES_MODELO].to_numpy()
    y = (df["nps_score"] <= cfg.DETRATOR_CUTOFF).astype(int).to_numpy()
    return df, X, y


def _proba_oof(X, y, seed=cfg.SEED):
    """Probabilidade de ser detrator, out-of-fold, com scaler fitado dentro do fold."""
    cv = StratifiedKFold(n_splits=cfg.CV_FOLDS, shuffle=True, random_state=seed)
    proba = np.zeros(len(y))
    for tr, va in cv.split(X, y):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xva = scaler.transform(X[va])
        model = RandomForestClassifier(**cfg.RF_PARAMS)
        model.fit(Xtr, y[tr])
        proba[va] = model.predict_proba(Xva)[:, 1]
    return proba


def _ece_mce(y_true, proba, n_bins=10):
    """Expected e Maximum Calibration Error com bins uniformes em [0, 1]."""
    bordas = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(proba, bordas[1:-1]), 0, n_bins - 1)
    ece = 0.0
    mce = 0.0
    linhas = []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        conf = float(proba[mask].mean())
        acc = float(y_true[mask].mean())
        gap = abs(acc - conf)
        ece += (n / len(proba)) * gap
        mce = max(mce, gap)
        linhas.append(dict(bin=b, faixa=[float(bordas[b]), float(bordas[b + 1])],
                           n=n, confianca=round(conf, 4), acerto=round(acc, 4),
                           gap=round(gap, 4)))
    return float(ece), float(mce), linhas


def _curva_confiabilidade(y_true, proba, n_bins=10):
    """Confianca media x acerto medio por decil de probabilidade (bins por quantil)."""
    df = pd.DataFrame({"p": proba, "y": y_true})
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    agg = df.groupby("bin", observed=True).agg(conf=("p", "mean"),
                                               acerto=("y", "mean"),
                                               n=("y", "size"))
    return agg["conf"].to_numpy(), agg["acerto"].to_numpy(), agg["n"].to_numpy()


def _ece_scorer_honesto(X, y, seed=cfg.SEED):
    """
    ECE do scorer recalibrado (StandardScaler + CalibratedClassifierCV) medido por
    CV externa: cada fold treina o scorer inteiro do zero e mede o ECE no fold de
    fora. Sem contaminacao in-sample. Retorna (ece_cru_oof, ece_scorer_oof).
    """
    cv = StratifiedKFold(cfg.CV_FOLDS, shuffle=True, random_state=seed)
    p_cru = np.zeros(len(y))
    p_cal = np.zeros(len(y))
    for tr, te in cv.split(X, y):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        rf = RandomForestClassifier(**cfg.RF_PARAMS).fit(Xtr, y[tr])
        p_cru[te] = rf.predict_proba(Xte)[:, 1]
        cal = CalibratedClassifierCV(
            RandomForestClassifier(**cfg.RF_PARAMS), method="isotonic",
            cv=StratifiedKFold(cfg.CV_FOLDS, shuffle=True, random_state=seed),
        ).fit(Xtr, y[tr])
        p_cal[te] = cal.predict_proba(Xte)[:, 1]
    ece_cru, _, _ = _ece_mce(y, p_cru)
    ece_cal, _, _ = _ece_mce(y, p_cal)
    return float(ece_cru), float(ece_cal)


def _grafico(y, proba_cru, proba_calib, ece_cru, ece_calib, brier, destino):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    conf_c, acerto_c, _ = _curva_confiabilidade(y, proba_cru)
    conf_r, acerto_r, _ = _curva_confiabilidade(y, proba_calib)

    ax1.plot([0, 1], [0, 1], "--", color="#888", label="calibracao perfeita")
    ax1.plot(conf_c, acerto_c, "o-", color="#e74c3c",
             label=f"modelo v1 cru  (ECE {ece_cru:.3f})")
    ax1.plot(conf_r, acerto_r, "s-", color="#2ecc71",
             label=f"recalibrado (isotonica)  (ECE {ece_calib:.3f})")
    ax1.set_xlabel("probabilidade prevista de ser detrator")
    ax1.set_ylabel("frequencia observada de detrator")
    ax1.set_title(f"Curva de confiabilidade  |  Brier cru = {brier:.3f}")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    ax2.hist(proba_calib[y == 1], bins=30, alpha=0.6, color="#d95f02",
             label="detrator real")
    ax2.hist(proba_calib[y == 0], bins=30, alpha=0.6, color="#1b9e77",
             label="nao detrator real")
    ax2.axvline(cfg.P_DETRATOR_ELEGIVEL, color="#2c3e50", ls=":",
                label=f"corte elegibilidade {cfg.P_DETRATOR_ELEGIVEL}")
    ax2.set_xlabel("probabilidade recalibrada de ser detrator")
    ax2.set_ylabel("clientes")
    ax2.set_title("Distribuicao das probabilidades recalibradas")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle(f"Modelo v1 - diagnostico de calibracao "
                 f"(dataset real da Fase 1, n={len(y)})", fontsize=12)
    fig.tight_layout()
    fig.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(fig)


def treinar_scorer(salvar=True):
    """
    Scorer de risco do experimento: StandardScaler + RF binario recalibrado por
    isotonica via CalibratedClassifierCV.

    CalibratedClassifierCV fita o RF em cada fold de CV e ajusta a isotonica sobre
    a saida held-out do fold, nunca in-sample. No predict, faz a media das copias
    calibradas por fold. Isso resolve o descasamento entre a probabilidade OOF (em
    que a isotonica aprende) e a probabilidade in-sample (com que um RF unico
    pontuaria), que inflava o ECE quando a isotonica era colada num RF final a mao.

    O scaler e fitado em todo o dataset de proposito: e artefato de deploy, nao ha
    holdout a proteger (a qualidade de calibracao e medida a parte por OOF em
    diagnosticar()). Persistido em cfg.SCORER_PATH.
    """
    _, X, y = _carregar_xy()

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    calibrado = CalibratedClassifierCV(
        RandomForestClassifier(**cfg.RF_PARAMS),
        method="isotonic",
        cv=StratifiedKFold(cfg.CV_FOLDS, shuffle=True, random_state=cfg.SEED),
    ).fit(Xs, y)

    bundle = {"scaler": scaler, "modelo": calibrado,
              "features": FEATURES_MODELO, "alvo": "detrator (nps_score <= 6)"}
    if salvar:
        cfg.SCORER_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, cfg.SCORER_PATH)
    return bundle


def carregar_scorer():
    if not cfg.SCORER_PATH.exists():
        return treinar_scorer(salvar=True)
    return joblib.load(cfg.SCORER_PATH)


def prever_risco(df_features, bundle=None):
    """P(Detrator) calibrada para um DataFrame que ja passou por criar_features()."""
    bundle = bundle or carregar_scorer()
    Xs = bundle["scaler"].transform(df_features[bundle["features"]].to_numpy())
    return bundle["modelo"].predict_proba(Xs)[:, 1]


def resumo_elegibilidade(df_features=None):
    """
    Sobre a probabilidade JA recalibrada (o que o experimento usa): quantos clientes
    passam no corte de elegibilidade e qual a densidade de detrator por estrato.
    """
    if df_features is None:
        df_features, _, _ = _carregar_xy()
    y = (df_features["nps_score"] <= cfg.DETRATOR_CUTOFF).astype(int).to_numpy()
    risco = prever_risco(df_features)

    elegivel = risco >= cfg.P_DETRATOR_ELEGIVEL
    frac_base = 2500 / len(y)  # dataset e ~1 mes; README usa 2500 pedidos/mes
    estratos = []
    for lo, hi in cfg.FAIXAS_P:
        m = elegivel & (risco >= lo) & (risco < hi)
        estratos.append({
            "faixa": [lo, hi],
            "n": int(m.sum()),
            "densidade_detrator": round(float(y[m].mean()), 4) if m.any() else None,
        })
    return {
        "corte": cfg.P_DETRATOR_ELEGIVEL,
        "n_elegivel": int(elegivel.sum()),
        "elegivel_por_mes": int(round(elegivel.sum() * frac_base)),
        "pct_da_base": round(float(elegivel.mean()), 4),
        "densidade_detrator": round(float(y[elegivel].mean()), 4) if elegivel.any() else None,
        "estratos": estratos,
    }


def diagnosticar(salvar=True, verbose=True):
    df, X, y = _carregar_xy()

    proba = _proba_oof(X, y)
    proba_calib = prever_risco(df)  # scorer recalibrado (o que o experimento usa)

    brier = brier_score_loss(y, proba)
    ece, mce, bins = _ece_mce(y, proba)
    ece_cru_oof, ece_scorer_oof = _ece_scorer_honesto(X, y)

    # veredito: ECE abaixo de 0,05 e a folga usual para tratar como probabilidade
    calibrado = ece < 0.05
    isotonica_ajuda = (ece_cru_oof - ece_scorer_oof) > 0.02

    # elegibilidade sobre a probabilidade JA recalibrada (o que o experimento usa)
    elgb = resumo_elegibilidade(df)

    resultado = {
        "rotulo": "diagnostico sobre dataset real da Fase 1",
        "n": int(len(y)),
        "taxa_detrator_base": round(float(y.mean()), 4),
        "brier": round(float(brier), 4),
        "ece": round(ece, 4),
        "mce": round(mce, 4),
        "ece_cru_cv_externa": round(ece_cru_oof, 4),
        "ece_scorer_recalibrado_cv_externa": round(ece_scorer_oof, 4),
        "elegibilidade_recalibrada": elgb,
        "veredito": {
            "calibrado_o_suficiente": bool(calibrado),
            "recalibracao_isotonica_ajuda": bool(isotonica_ajuda),
            "acao": (
                "usar P(Detrator) direto nos cortes de estrato"
                if calibrado else
                "recalibrar (isotonica) antes de fixar os cortes de estrato"
                if isotonica_ajuda else
                "P(Detrator) descalibrada e a isotonica ajuda pouco: tratar os "
                "cortes como ranking, nao como probabilidade, e definir os "
                "estratos por quantil de score"
            ),
        },
        "bins": bins,
    }

    if salvar:
        cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        _grafico(y, proba, proba_calib, ece_cru_oof, ece_scorer_oof, brier,
                 cfg.REPORTS_DIR / "reliability_v1.png")
        with open(cfg.REPORTS_DIR / "calibracao_v1.json", "w", encoding="utf-8") as f:
            json.dump(resultado, f, indent=2, ensure_ascii=False)

    if verbose:
        print("=" * 74)
        print("DIAGNOSTICO DE CALIBRACAO - modelo v1 (spec 0002 secao 0)")
        print("=" * 74)
        print(f"n = {resultado['n']}   taxa de detrator na base = "
              f"{resultado['taxa_detrator_base']:.1%}")
        print(f"Brier = {brier:.3f}   ECE = {ece:.3f}   MCE = {mce:.3f}   "
              f"(modelo v1 cru, OOF)")
        print(f"ECE por CV externa: cru {ece_cru_oof:.3f} -> "
              f"scorer recalibrado {ece_scorer_oof:.3f}")
        print("-" * 74)
        print(f"Elegibilidade (probabilidade recalibrada, corte "
              f">= {elgb['corte']}):")
        print(f"  {elgb['n_elegivel']} clientes ({elgb['pct_da_base']:.0%} da base), "
              f"~{elgb['elegivel_por_mes']}/mes, densidade de detrator "
              f"{elgb['densidade_detrator']:.1%}")
        for e in elgb["estratos"]:
            d = e["densidade_detrator"]
            print(f"  estrato [{e['faixa'][0]}, {e['faixa'][1]}): n={e['n']:4d}  "
                  f"densidade {d:.1%}" if d is not None else
                  f"  estrato [{e['faixa'][0]}, {e['faixa'][1]}): n=0")
        print("-" * 74)
        print(f"VEREDITO: {resultado['veredito']['acao']}")
        print("=" * 74)

    return resultado


if __name__ == "__main__":
    print("Treinando e salvando o scorer de risco recalibrado...")
    treinar_scorer(salvar=True)
    print(f"Scorer salvo em {cfg.SCORER_PATH.relative_to(cfg.RAIZ)}\n")
    diagnosticar()
