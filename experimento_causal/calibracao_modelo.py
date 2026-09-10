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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from utils import criar_features, FEATURES_MODELO
from experimento_causal import config as cfg


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


def _ganho_isotonico(proba, y, seed=cfg.SEED):
    """
    Quanto uma recalibracao isotonica reduziria o ECE, medido honestamente:
    fita a isotonica na metade A, avalia o ECE na metade B (nunca na mesma).
    """
    ia, ib = train_test_split(np.arange(len(y)), test_size=0.5,
                              random_state=seed, stratify=y)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(proba[ia], y[ia])
    proba_b_bruta = proba[ib]
    proba_b_calib = iso.predict(proba[ib])
    ece_antes, _, _ = _ece_mce(y[ib], proba_b_bruta)
    ece_depois, _, _ = _ece_mce(y[ib], proba_b_calib)
    return ece_antes, ece_depois


def _grafico(conf, acerto, proba, y_detrator, ece, brier, destino):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    ax1.plot([0, 1], [0, 1], "--", color="#888", label="calibracao perfeita")
    ax1.plot(conf, acerto, "o-", color="#2c7fb8", label="modelo v1 (OOF)")
    ax1.axvline(cfg.P_DETRATOR_OPERACAO, color="#e74c3c", ls=":",
                label=f"corte operacao {cfg.P_DETRATOR_OPERACAO}")
    ax1.axvline(cfg.P_DETRATOR_ELEGIVEL, color="#2ecc71", ls=":",
                label=f"corte elegibilidade {cfg.P_DETRATOR_ELEGIVEL}")
    ax1.set_xlabel("probabilidade prevista de ser detrator")
    ax1.set_ylabel("frequencia observada de detrator")
    ax1.set_title(f"Curva de confiabilidade  |  Brier={brier:.3f}  ECE={ece:.3f}")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    ax2.hist(proba[y_detrator == 1], bins=30, alpha=0.6, color="#d95f02",
             label="detrator real")
    ax2.hist(proba[y_detrator == 0], bins=30, alpha=0.6, color="#1b9e77",
             label="nao detrator real")
    ax2.axvline(cfg.P_DETRATOR_ELEGIVEL, color="#2ecc71", ls=":")
    ax2.set_xlabel("probabilidade prevista de ser detrator")
    ax2.set_ylabel("clientes")
    ax2.set_title("Distribuicao das probabilidades previstas")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle("Modelo v1 - diagnostico de calibracao (dataset real da Fase 1)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(fig)


def diagnosticar(salvar=True, verbose=True):
    df = pd.read_csv(cfg.DATA_PATH).drop(columns=cfg.LEAKAGE_COLS)
    df = criar_features(df)
    X = df[FEATURES_MODELO].to_numpy()
    y = (df["nps_score"] <= cfg.DETRATOR_CUTOFF).astype(int).to_numpy()

    proba = _proba_oof(X, y)

    brier = brier_score_loss(y, proba)
    ece, mce, bins = _ece_mce(y, proba)
    conf, acerto, _ = _curva_confiabilidade(y, proba)
    ece_iso_antes, ece_iso_depois = _ganho_isotonico(proba, y)

    # densidade de detrator dentro do corte de elegibilidade
    mask_elegivel = proba >= cfg.P_DETRATOR_ELEGIVEL
    densidade_elegivel = float(y[mask_elegivel].mean()) if mask_elegivel.any() else float("nan")

    # veredito: ECE abaixo de 0,05 e a folga usual para tratar como probabilidade
    calibrado = ece < 0.05
    isotonica_ajuda = (ece_iso_antes - ece_iso_depois) > 0.02

    resultado = {
        "rotulo": "diagnostico sobre dataset real da Fase 1",
        "n": int(len(y)),
        "taxa_detrator_base": round(float(y.mean()), 4),
        "brier": round(float(brier), 4),
        "ece": round(ece, 4),
        "mce": round(mce, 4),
        "corte_elegibilidade": cfg.P_DETRATOR_ELEGIVEL,
        "clientes_no_corte": int(mask_elegivel.sum()),
        "densidade_detrator_no_corte": round(densidade_elegivel, 4),
        "isotonica_ece_antes": round(ece_iso_antes, 4),
        "isotonica_ece_depois": round(ece_iso_depois, 4),
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
        _grafico(conf, acerto, proba, y, ece, brier,
                 cfg.REPORTS_DIR / "reliability_v1.png")
        with open(cfg.REPORTS_DIR / "calibracao_v1.json", "w", encoding="utf-8") as f:
            json.dump(resultado, f, indent=2, ensure_ascii=False)

    if verbose:
        print("=" * 74)
        print("DIAGNOSTICO DE CALIBRACAO - modelo v1 (spec 0002 secao 0)")
        print("=" * 74)
        print(f"n = {resultado['n']}   taxa de detrator na base = "
              f"{resultado['taxa_detrator_base']:.1%}")
        print(f"Brier = {brier:.3f}   ECE = {ece:.3f}   MCE = {mce:.3f}")
        print(f"No corte P >= {cfg.P_DETRATOR_ELEGIVEL}: "
              f"{resultado['clientes_no_corte']} clientes, "
              f"{densidade_elegivel:.1%} sao detrator de fato")
        print(f"ECE com recalibracao isotonica: {ece_iso_antes:.3f} -> "
              f"{ece_iso_depois:.3f} (split separado)")
        print("-" * 74)
        print(f"VEREDITO: {resultado['veredito']['acao']}")
        print("=" * 74)

    return resultado


if __name__ == "__main__":
    diagnosticar()
