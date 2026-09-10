"""
Monitor de Data Drift — Tech Challenge NPS Fase 1 (Ticket 0004)

Substitui a versao decorativa anterior (threshold arbitrario numa unica
feature, sem teste estatistico real — o comentario original admitia
"apenas demonstraremos a estrutura do monitor").

Metodo: Kolmogorov-Smirnov two-sample test comparando a distribuicao de
cada feature numerica no lote novo contra a amostra de referencia do
treino (`models/v1/train_reference_sample.csv` = o X_train exato em que o
modelo foi treinado, escrito por train_pipeline.py apos o split).

Como o teste roda em ~20 features simultaneamente informando a mesma
decisao (retreinar ou nao), aplica correcao de comparacoes multiplas
(Holm, por padrao) antes de reportar drift — sem isso, ~1 feature em 20
"detectaria" drift so por ruido amostral (alpha=0.05 nominal).
"""
import sys
import json
import argparse
from datetime import datetime, timezone

import pandas as pd
from scipy.stats import ks_2samp
from statsmodels.stats.multitest import multipletests

from utils import criar_features, FEATURES_MODELO

REFERENCE_PATH = 'models/v1/train_reference_sample.csv'


def check_drift(new_data_path, reference_path=REFERENCE_PATH, alpha=0.05,
                 correction_method='holm'):
    """
    Compara a distribuicao de cada feature numerica do lote novo contra a
    amostra de referencia do treino via KS-test, com correcao de
    comparacoes multiplas.

    Retorna um dict com o relatorio completo (tambem imprime resumo no
    console). Levanta FileNotFoundError se algum dos dois arquivos nao
    existir — falha explicita, nao retorno silencioso de None.
    """
    reference = pd.read_csv(reference_path)

    new_df = pd.read_csv(new_data_path)
    new_df = criar_features(new_df)

    features_numericas = [
        f for f in FEATURES_MODELO
        if f in reference.columns and f in new_df.columns
    ]

    resultados = []
    for feature in features_numericas:
        ref_values = reference[feature].dropna()
        new_values = new_df[feature].dropna()
        stat, p_value = ks_2samp(ref_values, new_values)
        resultados.append({
            'feature': feature,
            'ks_statistic': float(stat),
            'p_value': float(p_value),
            'ref_mean': float(ref_values.mean()),
            'new_mean': float(new_values.mean()),
        })

    p_values = [r['p_value'] for r in resultados]
    rejected, p_corrected, _, _ = multipletests(
        p_values, alpha=alpha, method=correction_method
    )

    for r, rej, p_corr in zip(resultados, rejected, p_corrected):
        r['p_value_corrigido'] = float(p_corr)
        r['drift_detectado'] = bool(rej)

    n_drift = sum(r['drift_detectado'] for r in resultados)

    relatorio = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'new_data_path': new_data_path,
        'reference_path': reference_path,
        'n_reference': len(reference),
        'n_new': len(new_df),
        'alpha': alpha,
        'correction_method': correction_method,
        'n_features_testadas': len(features_numericas),
        'n_features_com_drift': n_drift,
        'resultados': sorted(resultados, key=lambda r: r['p_value_corrigido']),
    }

    _print_resumo(relatorio)
    return relatorio


def _print_resumo(relatorio):
    print("=" * 80)
    print("MONITOR DE DATA DRIFT — KS-test com correcao de comparacoes multiplas")
    print("=" * 80)
    print(f"Referencia: {relatorio['n_reference']} amostras ({relatorio['reference_path']})")
    print(f"Lote novo:  {relatorio['n_new']} amostras ({relatorio['new_data_path']})")
    print(f"Features testadas: {relatorio['n_features_testadas']}")
    print(f"Correcao: {relatorio['correction_method']} (alpha={relatorio['alpha']})")
    print()

    if relatorio['n_features_com_drift'] == 0:
        print("Nenhuma feature com drift estatisticamente significativo apos correcao.")
    else:
        print(f"ALERTA: {relatorio['n_features_com_drift']} feature(s) com drift detectado:")
        for r in relatorio['resultados']:
            if r['drift_detectado']:
                print(
                    f"  - {r['feature']}: p_corrigido={r['p_value_corrigido']:.4f} "
                    f"(ref_mean={r['ref_mean']:.2f}, new_mean={r['new_mean']:.2f})"
                )
        print()
        print("Recomendado: investigar causa do drift antes de decidir re-treinamento.")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor de data drift via KS-test")
    parser.add_argument("new_data_path", help="CSV com o lote novo de dados")
    parser.add_argument("--reference", default=REFERENCE_PATH,
                         help=f"CSV de referencia do treino (default: {REFERENCE_PATH})")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--method", default="holm",
                         help="Metodo de correcao: holm, bonferroni, fdr_bh, etc.")
    parser.add_argument("--output", help="Salvar relatorio completo em JSON neste caminho")
    args = parser.parse_args()

    relatorio = check_drift(args.new_data_path, args.reference, args.alpha, args.method)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(relatorio, f, indent=2, ensure_ascii=False)
        print(f"\nRelatorio completo salvo em {args.output}")

    sys.exit(1 if relatorio['n_features_com_drift'] > 0 else 0)
