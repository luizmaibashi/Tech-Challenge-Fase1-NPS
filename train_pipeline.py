import pandas as pd
import joblib
import json
import os
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from utils import criar_features, FEATURES_MODELO

# Configurações
DATA_PATH = 'data/desafio_nps_fase_1.csv'
MODEL_VERSION = 'v1'
MODEL_DIR = f'models/{MODEL_VERSION}'

# Leakage: variáveis determinadas depois do momento da predição (ver
# reports/dicionario_desafio_nps.md). Removidas antes de qualquer feature.
LEAKAGE_COLS = ['repeat_purchase_30d', 'csat_internal_score']

def train():
    print(f"Iniciando treinamento do NPS Predictor {MODEL_VERSION}...")
    
    # 1. Carregamento
    if not os.path.exists(DATA_PATH):
        print(f"Erro: Arquivo de dados não encontrado em {DATA_PATH}")
        return
    
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=LEAKAGE_COLS)

    # 2. Pre-processamento de Target (Label Encoding manual para classes NPS)
    # Classificação NPS clássica sobre nota contínua: Detrator (<= 6) -> 0,
    # Neutro (7-8) -> 1, Promotor (>= 9) -> 2. O corte de Detrator é <= 6
    # (não < 7): a faixa 6,01-6,99 é Neutro, coerente com o padrão de mercado
    # aplicado à nota inteira. Mesma regra em benchmark_modelos.py,
    # threshold_calibration.py e reports/dicionario_desafio_nps.md.
    def categorize_nps(score):
        if score <= 6: return 0
        elif score <= 8: return 1
        else: return 2

    df['target'] = df['nps_score'].apply(categorize_nps)

    # 3. Feature Engineering
    df = criar_features(df)

    # 4. Split de Dados
    X = df[FEATURES_MODELO]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 5. Construção do Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=100, 
            max_depth=7, 
            random_state=42, 
            class_weight='balanced', 
            n_jobs=-1
        ))
    ])
    
    # 6. Treinamento
    pipeline.fit(X_train, y_train)
    
    # 7. Avaliação (teste consultado uma única vez, aqui)
    y_pred = pipeline.predict(X_test)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"Treinamento concluído! F1-Score Macro: {f1_macro:.4f}")
    print(f"Recall Detrator (holdout): {report['0']['recall']:.4f}")

    # 8. Salvamento e Versionamento
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = f"{MODEL_DIR}/pipeline_completo.pkl"
    joblib.dump(pipeline, model_path)

    # Baseline do monitor de drift: as features exatas em que o modelo foi
    # treinado (X_train pós-split, pós-remoção de leakage). monitor.py compara
    # a distribuição de lotes novos contra este arquivo.
    X_train.to_csv(f"{MODEL_DIR}/train_reference_sample.csv", index=False)

    metadata = {
        "version": MODEL_VERSION,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target_definition": "Detrator nps_score <= 6 | Neutro 7-8 | Promotor >= 9",
        "n_train": len(X_train),
        "n_test": len(X_test),
        "metrics": {
            "f1_macro": f1_macro,
            "recall_detrator": report['0']['recall']
        },
        "parameters": {
            "n_estimators": 100,
            "max_depth": 7,
            "class_weight": "balanced"
        },
        "features": FEATURES_MODELO
    }

    with open(f"{MODEL_DIR}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Modelo, metadados e baseline de drift salvos em {MODEL_DIR}/")

if __name__ == "__main__":
    train()
