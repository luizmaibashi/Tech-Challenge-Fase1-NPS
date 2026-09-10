"""
Threshold Calibrado por Custo de Negocio — Tech Challenge NPS Fase 1 (Ticket 0003)

Diferente da classificacao multiclasse do benchmark (Detrator/Neutro/Promotor,
decidida por argmax de probabilidade — ver benchmark_modelos.py), este script
resolve uma pergunta de negocio distinta: "a partir de qual probabilidade de
ser Detrator vale a pena disparar a acao profilatica (cupom, CS VIP)?"

Gate ML da base (threshold != balanceamento de treino): o RF ja usa
class_weight='balanced' no treino — isso corrige a proporcao de classes na
funcao de perda, mas nao decide o corte de probabilidade na inferencia. As
probabilidades usadas aqui sao OOF (out-of-fold, geradas por CV) para nao
calibrar o threshold sobre dado que influenciou o proprio treino.

Matriz de custo (premissas do README, secao 5 - ROI):
- Custo de ACAO (cupom/CS VIP) por Falso Positivo: R$ 30,00
- Custo de OMISSAO por Falso Negativo: taxa_retencao x LTV = 0.35 x R$350
  = R$ 122,50 (oportunidade de retencao perdida ao nao agir sobre um
  Detrator real)
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

from utils import criar_features, FEATURES_MODELO

CUSTO_CUPOM = 30.0
TAXA_RETENCAO = 0.35
LTV_RETIDO = 350.0
CUSTO_OPORTUNIDADE_FN = TAXA_RETENCAO * LTV_RETIDO  # R$ 122.50


def main():
    print("=" * 80)
    print("THRESHOLD CALIBRADO POR CUSTO — Classe Detrator (acao de retencao)")
    print("=" * 80)

    df = pd.read_csv('data/desafio_nps_fase_1.csv')
    df_clean = df.drop(columns=['repeat_purchase_30d', 'csat_internal_score'])
    df_features = criar_features(df_clean)

    X = df_features[FEATURES_MODELO].to_numpy()
    y_detrator = (df_features['nps_score'] <= 6).astype(int)

    print(f"\nGerando probabilidades OOF (out-of-fold) via CV 5-fold...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proba_oof = np.zeros(len(y_detrator))

    for train_idx, val_idx in cv.split(X, y_detrator):
        # Scaler fitado só no train do fold — a probabilidade OOF não pode
        # ver a distribuição da própria validação, nem via normalização.
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])

        model = RandomForestClassifier(
            n_estimators=100, max_depth=7, class_weight='balanced',
            random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_detrator.iloc[train_idx])
        proba_oof[val_idx] = model.predict_proba(X_val)[:, 1]

    print(f"Custo por Falso Positivo (acao desnecessaria): R$ {CUSTO_CUPOM:.2f}")
    print(f"Custo por Falso Negativo (oportunidade perdida): R$ {CUSTO_OPORTUNIDADE_FN:.2f}")
    print(f"Razao de custo FN/FP: {CUSTO_OPORTUNIDADE_FN/CUSTO_CUPOM:.2f}x")

    # Grid de thresholds
    thresholds_grid = np.arange(0.02, 0.98, 0.01)
    resultados = []
    for t in thresholds_grid:
        pred = (proba_oof >= t).astype(int)
        fp = int(((pred == 1) & (y_detrator == 0)).sum())
        fn = int(((pred == 0) & (y_detrator == 1)).sum())
        tp = int(((pred == 1) & (y_detrator == 1)).sum())
        tn = int(((pred == 0) & (y_detrator == 0)).sum())
        custo_total = fp * CUSTO_CUPOM + fn * CUSTO_OPORTUNIDADE_FN
        resultados.append({
            'threshold': round(float(t), 2), 'fp': fp, 'fn': fn, 'tp': tp, 'tn': tn,
            'precision': precision_score(y_detrator, pred, zero_division=0),
            'recall': recall_score(y_detrator, pred, zero_division=0),
            'f1': f1_score(y_detrator, pred, zero_division=0),
            'custo_total': float(custo_total),
        })

    df_res = pd.DataFrame(resultados)
    idx_otimo = df_res['custo_total'].idxmin()
    otimo = df_res.loc[idx_otimo]

    idx_padrao = (df_res['threshold'] - 0.5).abs().idxmin()
    padrao = df_res.loc[idx_padrao]

    fator_escala = 2500 / len(y_detrator)  # README usa 2500 pedidos/mes
    economia_mensal = (padrao['custo_total'] - otimo['custo_total']) * fator_escala

    print("\n" + "=" * 80)
    print("RESULTADO")
    print("=" * 80)
    print(f"\n[THRESHOLD PADRAO = 0.50]")
    print(f"  FP={int(padrao['fp'])}  FN={int(padrao['fn'])}  Recall={padrao['recall']:.3f}  "
          f"Custo total={padrao['custo_total']:.2f}")

    print(f"\n[THRESHOLD OTIMO = {otimo['threshold']:.2f}]")
    print(f"  FP={int(otimo['fp'])}  FN={int(otimo['fn'])}  Recall={otimo['recall']:.3f}  "
          f"Custo total={otimo['custo_total']:.2f}")

    print(f"\nEconomia estimada: R$ {economia_mensal:.2f}/mes (escala de 2.500 pedidos/mes)")
    print(f"Recall com threshold otimo: {otimo['recall']:.1%} "
          f"(meta do PROBLEM.md: >= 75%, {'ATINGIDA' if otimo['recall']>=0.75 else 'NAO ATINGIDA'})")

    # Grafico: custo total vs threshold
    plt.figure(figsize=(10, 6))
    plt.plot(df_res['threshold'], df_res['custo_total'], color='#3498db', linewidth=2)
    plt.axvline(otimo['threshold'], color='#2ecc71', linestyle='--',
                label=f"Ótimo = {otimo['threshold']:.2f} (custo mín.)")
    plt.axvline(0.5, color='#e74c3c', linestyle=':',
                label=f"Padrão = 0.50")
    plt.xlabel('Threshold de decisão (P(Detrator) >= threshold → ação)')
    plt.ylabel('Custo total esperado (R$)')
    plt.title('Custo de Negócio vs Threshold de Decisão\n'
               f'(Custo FP = R\\${CUSTO_CUPOM:.0f}  |  Custo FN = R\\${CUSTO_OPORTUNIDADE_FN:.2f})')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('reports/threshold_custo.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nGráfico salvo em reports/threshold_custo.png")

    # Salvar relatorio completo
    relatorio = {
        'premissas': {
            'custo_cupom': CUSTO_CUPOM,
            'taxa_retencao': TAXA_RETENCAO,
            'ltv_retido': LTV_RETIDO,
            'custo_oportunidade_fn': CUSTO_OPORTUNIDADE_FN,
        },
        'threshold_padrao': padrao.to_dict(),
        'threshold_otimo': otimo.to_dict(),
        'economia_mensal_estimada': float(economia_mensal),
        'meta_recall_problem_md': 0.75,
        'meta_atingida': bool(otimo['recall'] >= 0.75),
    }
    with open('reports/threshold_calibration.json', 'w', encoding='utf-8') as f:
        json.dump(relatorio, f, indent=2, ensure_ascii=False)
    print("Relatório completo salvo em reports/threshold_calibration.json")

    df_res.to_csv('reports/threshold_grid.csv', index=False)
    print("Grid completo salvo em reports/threshold_grid.csv")


if __name__ == "__main__":
    main()
