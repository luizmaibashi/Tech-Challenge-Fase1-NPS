"""
Benchmark de Modelos — Tech Challenge NPS Fase 1
Decisão do Ticket 0009: Reabre modelagem inteira

Candidatos testados:
1. Random Forest (modelo em produção — models/v1/pipeline_completo.pkl)
2. Gradient Boosting (sklearn GradientBoostingClassifier)
3. Logistic Regression (baseline linear)

CV: 5-fold stratified (respeita desbalanceamento 74% Detrator).
Métrica principal: F1-Score macro (não acurácia).

Feature set: exatamente as 20 FEATURES_MODELO de utils.py — o mesmo conjunto
que api.py, app/deploy.py e train_pipeline.py usam. Região NÃO entra (o
pipeline de produção não usa; incluí-la aqui compararia um modelo diferente
do que está no ar).

Scaler: fitado dentro de cada fold (só no train), nunca sobre o dataset
inteiro — senão a normalização vaza a distribuição da validação.

Output:
- reports/benchmark_results.csv (resumo de CV)
- reports/cv_scores.csv (fold-by-fold)
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

# Importar função de feature engineering centralizada
from utils import criar_features, FEATURES_MODELO

LEAKAGE_COLS = ['repeat_purchase_30d', 'csat_internal_score']

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
    df_clean = df.drop(columns=LEAKAGE_COLS)
    print(f"   Colunas restantes: {df_clean.shape[1]}")

    # 3. Feature engineering
    print("\n[3/5] Aplicando feature engineering (utils.py:criar_features)...")
    df_features = criar_features(df_clean)

    # Feature set = exatamente o de produção (20 colunas, sem região).
    X = df_features[FEATURES_MODELO].copy()

    # Classificação NPS: Detrator <= 6 | Neutro 7-8 | Promotor >= 9
    # (mesma regra de train_pipeline.py e threshold_calibration.py)
    def nps_category(score):
        if score <= 6:
            return 'Detrator'
        elif score <= 8:
            return 'Neutro'
        else:
            return 'Promotor'

    y = df_features['nps_score'].apply(nps_category)

    print(f"   Features: {X.shape[1]} ({', '.join(FEATURES_MODELO[:4])}...)")
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

    # 5. Rodar benchmark
    print("\n[5/5] Rodando cross-validation e benchmark...\n")

    results_cv = []
    final_results = []

    for model_name, model in models.items():
        print(f"   [{model_name}] CV em progresso...", end=' ', flush=True)

        fold_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Scaler fitado só no train do fold (sem vazar a validação)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

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
    print(f"Modelo em produção: Random Forest (models/v1/pipeline_completo.pkl).")
    print(f"Threshold de decisão calibrado à parte: threshold_calibration.py.")
    print("="*80)

if __name__ == '__main__':
    main()
