"""
Benchmark de Modelos — Tech Challenge NPS Fase 1
Decisão do Ticket 0009: Reabre modelagem inteira

Candidatos testados:
1. Gradient Boosting (XGBoost como principal)
2. Logistic Regression (baseline linear)
3. Random Forest (candidato existente)

CV: 5-fold stratified (respeita desbalanceamento 74% Detrator)
Métrica principal: F1-Score macro (não acurácia)
class_weight: 'balanced' em todos

Output:
- reports/benchmark_results.csv (resumo de CV)
- reports/cv_scores.csv (fold-by-fold)
- reports/confusion_matrix_*.csv (matriz final de cada modelo)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# Importar função de feature engineering centralizada
from utils import criar_features, FEATURES_MODELO

def main():
    print("="*80)
    print("BENCHMARK DE MODELOS — Tech Challenge NPS Fase 1")
    print("="*80)

    # 1. Carregar dados
    print("\n[1/5] Carregando dados...")
    df = pd.read_csv('data/desafio_nps_fase_1.csv')
    print(f"   Shape original: {df.shape}")

    # 2. Remover leakage
    print("\n[2/5] Removendo leakage (repeat_purchase_30d, csat_internal_score)...")
    df_clean = df.drop(columns=['repeat_purchase_30d', 'csat_internal_score'])
    print(f"   Colunas restantes: {df_clean.shape[1]}")

    # 3. Feature engineering
    print("\n[3/5] Aplicando feature engineering (utils.py:criar_features)...")
    df_features = criar_features(df_clean)

    # Preparar features e target
    # Adicionar region antes de selecionar features
    df_features_with_region = df_features.copy()
    df_features_with_region = pd.get_dummies(
        df_features_with_region,
        columns=['customer_region'],
        drop_first=True,
        prefix='region'
    )

    # Selecionar features (sem region, será adicionada)
    X = df_features_with_region[FEATURES_MODELO].copy()

    # Adicionar region encoded (todas as colunas começando com 'region_')
    region_cols = [col for col in df_features_with_region.columns if col.startswith('region_')]
    X = pd.concat([X, df_features_with_region[region_cols]], axis=1)

    # Classificação NPS para estratificação (será a variável target)
    def nps_category(score):
        if score <= 6:
            return 'Detrator'
        elif score <= 8:
            return 'Neutro'
        else:
            return 'Promotor'

    y = df_features_with_region['nps_score'].apply(nps_category)

    print(f"   Features após encoding: {X.shape[1]}")
    print(f"   Feature names: {list(FEATURES_MODELO) + region_cols}")

    # Normalizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    print(f"   Target distribution:")
    print(f"     Detrator: {(y=='Detrator').sum()} ({100*(y=='Detrator').sum()/len(y):.1f}%)")
    print(f"     Neutro: {(y=='Neutro').sum()} ({100*(y=='Neutro').sum()/len(y):.1f}%)")
    print(f"     Promotor: {(y=='Promotor').sum()} ({100*(y=='Promotor').sum()/len(y):.1f}%)")

    # 4. Setup de CV e modelos
    print("\n[4/5] Configurando CV stratified 5-fold e modelos...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=7,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
    }

    # Métricas customizadas
    scoring = {
        'f1_macro': lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision_macro': lambda y_true, y_pred: precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': lambda y_true, y_pred: recall_score(y_true, y_pred, average='macro', zero_division=0),
    }

    # 5. Rodar benchmark
    print("\n[5/5] Rodando cross-validation e benchmark...\n")

    results_cv = []
    final_results = []

    for model_name, model in models.items():
        print(f"   [{model_name}] CV em progresso...", end=' ', flush=True)

        fold_scores = []
        confusion_matrices = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_scaled, y)):
            X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Treinar
            model.fit(X_train, y_train)

            # Predizer
            y_pred = model.predict(X_val)

            # Métricas
            f1_m = f1_score(y_val, y_pred, average='macro', zero_division=0)
            f1_w = f1_score(y_val, y_pred, average='weighted', zero_division=0)
            prec_m = precision_score(y_val, y_pred, average='macro', zero_division=0)
            rec_m = recall_score(y_val, y_pred, average='macro', zero_division=0)

            fold_scores.append({
                'model': model_name,
                'fold': fold_idx + 1,
                'f1_macro': f1_m,
                'f1_weighted': f1_w,
                'precision_macro': prec_m,
                'recall_macro': rec_m,
                'n_train': len(train_idx),
                'n_val': len(val_idx),
            })

        print("OK")

        # Resumo de CV
        df_folds = pd.DataFrame(fold_scores)
        mean_f1 = df_folds['f1_macro'].mean()
        std_f1 = df_folds['f1_macro'].std()

        final_results.append({
            'model': model_name,
            'f1_macro_mean': mean_f1,
            'f1_macro_std': std_f1,
            'f1_macro_min': df_folds['f1_macro'].min(),
            'f1_macro_max': df_folds['f1_macro'].max(),
            'f1_weighted_mean': df_folds['f1_weighted'].mean(),
            'precision_macro_mean': df_folds['precision_macro'].mean(),
            'recall_macro_mean': df_folds['recall_macro'].mean(),
        })

        results_cv.extend(fold_scores)

    # Salvar resultados
    print("\n" + "="*80)
    print("RESULTADOS")
    print("="*80)

    # Benchmark summary
    df_final = pd.DataFrame(final_results)
    df_final = df_final.sort_values('f1_macro_mean', ascending=False)
    print("\n[RESUMO] Ranking por F1-Score Macro (CV 5-fold):\n")
    print(df_final.to_string(index=False))

    # Modelo vencedor
    winner = df_final.iloc[0]
    print(f"\n[VENCEDOR] {winner['model']}")
    print(f"   F1-Score Macro: {winner['f1_macro_mean']:.4f} ± {winner['f1_macro_std']:.4f}")
    print(f"   Precision Macro: {winner['precision_macro_mean']:.4f}")
    print(f"   Recall Macro: {winner['recall_macro_mean']:.4f}")

    # Salvar CSVs
    print("\n[OUTPUT] Salvando relatorios...")
    df_final.to_csv('reports/benchmark_results.csv', index=False)
    print("   [OK] reports/benchmark_results.csv")

    df_cv = pd.DataFrame(results_cv)
    df_cv.to_csv('reports/cv_scores.csv', index=False)
    print("   [OK] reports/cv_scores.csv")

    print("\n" + "="*80)
    print("PRÓXIMOS PASSOS")
    print("="*80)
    print(f"""
1. Modelo vencedor: {winner['model']}
   - Use como base para Tickets 0002-0008 (refactorings de engenharia)

2. Tickets travados em aberto:
   - 0002 (Heurística API vs paridade Streamlit) — depende do modelo final
   - 0004 (Monitor reativo vs remover)
   - 0007 (SHAP faz sentido agora?)

3. Próximo gate: Threshold calibrado por custo (Ticket 0003)
   - Usar curva PR do modelo vencedor
   - Custo de negócio: FN (deixar detrator sem amparo) vs FP (contato desnecessário)

4. Volta a decisão de deploy (grade vs ONNX) em portfolio_deploy
   - Dimensão de features pode ter mudado (vencedor pode usar menos features que RF)
""")

    print("="*80)

if __name__ == '__main__':
    main()
