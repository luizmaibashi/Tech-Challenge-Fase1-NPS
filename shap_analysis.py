"""
Explicabilidade via SHAP — Tech Challenge NPS Fase 1 (Ticket 0007)

Corrige a lacuna entre o README/DIAGNOSTICO_REFAT.md (que alegava "Fase 4
CONCLUIDO" com este script) e a realidade (so existia Gini importance
nativa da RF — importancia global, nao explica predicao individual).

Escopo deliberadamente enxuto (decisao registrada no Ticket 0007): o
README do projeto ja fecha o escopo de modelagem como "complemento
analitico, nao eixo principal" — este script entrega o minimo que resolve
a lacuna real (explicacao por instancia), sem replicar o pipeline completo
de SHAP do projeto irmao (pipeline_churn_finance), que nao tem consumidor
aqui.

Gera dois artefatos:
- reports/shap_summary.png — importancia global COM direcao do efeito
  (diferente de Gini, que so da magnitude)
- reports/shap_waterfall_detrator.png — explicacao de UM cliente Detrator
  real: quais features empurraram a predicao pra essa classe
"""
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from utils import criar_features, FEATURES_MODELO

MODEL_PATH = 'models/v1/pipeline_completo.pkl'
DATA_PATH = 'data/desafio_nps_fase_1.csv'
OUTPUT_DIR = 'reports'


def main():
    print("Carregando modelo e dados...")
    pipeline = joblib.load(MODEL_PATH)
    scaler = pipeline.named_steps['scaler']
    clf = pipeline.named_steps['clf']

    df = pd.read_csv(DATA_PATH)
    df = criar_features(df)
    X = df[FEATURES_MODELO]

    def nps_category(score):
        if score <= 6:
            return 0  # Detrator
        elif score <= 8:
            return 1  # Neutro
        return 2  # Promotor

    y = df['nps_score'].apply(nps_category)

    # SHAP explica o modelo sobre os dados como ele de fato os recebe —
    # apos o scaler, igual ao que acontece dentro do Pipeline em produção.
    X_scaled = scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURES_MODELO, index=X.index)

    print("Calculando SHAP values (TreeExplainer, rapido para Random Forest)...")
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_scaled_df)

    # shap_values multiclasse: lista de arrays (um por classe) ou array 3D
    # dependendo da versao do shap instalada — normaliza para lista.
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = [shap_values[:, :, c] for c in range(shap_values.shape[2])]

    classes = ['Detrator', 'Neutro', 'Promotor']

    # --- 1. Summary plot: importancia global COM direcao (classe Detrator,
    #    a classe de acao — interessa saber o que EMPURRA pra detratacao) ---
    print("Gerando summary plot (classe Detrator)...")
    plt.figure()
    shap.summary_plot(
        shap_values[0], X_scaled_df, show=False,
        plot_size=(10, 8)
    )
    plt.title("SHAP Summary — Impacto das Features na Predição de Detrator")
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Salvo em {OUTPUT_DIR}/shap_summary.png")

    # --- 2. Waterfall de 1 cliente Detrator real: explicacao individual ---
    detrator_idx = y[y == 0].index[0]
    pos = X.index.get_loc(detrator_idx)

    print(f"\nExemplo de explicação individual (customer_id={df.loc[detrator_idx, 'customer_id']}):")
    print(f"  NPS real: {df.loc[detrator_idx, 'nps_score']:.1f} (Detrator)")
    print(f"  delivery_delay_days={df.loc[detrator_idx, 'delivery_delay_days']}, "
          f"complaints_count={df.loc[detrator_idx, 'complaints_count']}, "
          f"customer_service_contacts={df.loc[detrator_idx, 'customer_service_contacts']}")

    expected_value = explainer.expected_value[0]
    explanation = shap.Explanation(
        values=shap_values[0][pos],
        base_values=expected_value,
        data=X_scaled_df.iloc[pos].values,
        feature_names=FEATURES_MODELO,
    )

    plt.figure()
    shap.waterfall_plot(explanation, show=False, max_display=10)
    plt.title(f"Por que este cliente foi classificado como Detrator?")
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/shap_waterfall_detrator.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Salvo em {OUTPUT_DIR}/shap_waterfall_detrator.png")

    # --- Resumo em texto: top 5 features por importancia media |SHAP| ---
    mean_abs_shap = np.abs(shap_values[0]).mean(axis=0)
    top5_idx = np.argsort(mean_abs_shap)[::-1][:5]
    print("\nTop 5 features (importância média |SHAP|, classe Detrator):")
    for i in top5_idx:
        print(f"  {FEATURES_MODELO[i]}: {mean_abs_shap[i]:.4f}")


if __name__ == "__main__":
    main()
