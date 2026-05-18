import pandas as pd
import joblib
import os
from utils import criar_features, FEATURES_MODELO

# Configurações Stanford Lifecycle
DATA_PATH = 'data/desafio_nps_fase_1.csv'
MODEL_PATH = 'models/v1/pipeline_completo.pkl'
OUTPUT_REPORT = 'reports/manual_error_analysis_sample.csv'

def generate_error_analysis_sample():
    print("Iniciando geração de amostra para Manual Error Analysis (Stanford Method)...")
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
        print("Erro: Modelo ou base de dados não encontrados.")
        return
        
    df = pd.read_csv(DATA_PATH)
    pipeline = joblib.load(MODEL_PATH)
    
    # 1. Transformar target original
    def categorize_nps(score):
        if score <= 6: return 0 # Detrator
        elif score <= 8: return 1 # Neutro
        else: return 2 # Promotor
        
    df['target_real'] = df['nps_score'].apply(categorize_nps)
    
    # 2. Pipeline e features
    df_feat = criar_features(df)
    X = df_feat[FEATURES_MODELO]
    
    # 3. Predição
    df['target_predito'] = pipeline.predict(X)
    
    # 4. Filtrar Erros Críticos (Falso Negativo de Detrator é o pior erro)
    # Cliente era Detrator (0) mas previmos Neutro(1) ou Promotor(2)
    erros = df[df['target_real'] != df['target_predito']].copy()
    
    # Adicionar colunas descritivas para facilitar leitura visual
    map_classes = {0: "Detrator", 1: "Neutro", 2: "Promotor"}
    erros['Real_Class'] = erros['target_real'].map(map_classes)
    erros['Pred_Class'] = erros['target_predito'].map(map_classes)
    
    # Pegar uma amostra de ~100 erros focada em Detratores mascarados
    falsos_positivos = erros[erros['target_real'] == 0]
    
    amostra_tamanho = min(100, len(falsos_positivos))
    amostra_analise = falsos_positivos.sample(n=amostra_tamanho, random_state=42)
    
    # Selecionar colunas chave para a análise manual
    colunas_foco = [
        'Real_Class', 'Pred_Class', 'nps_score', 'delivery_delay_days', 
        'customer_service_contacts', 'complaints_count', 'order_value', 'freight_value'
    ]
    
    os.makedirs('reports', exist_ok=True)
    amostra_analise[colunas_foco].to_csv(OUTPUT_REPORT, index=False)
    
    print(f"Sucesso! Amostra de {amostra_tamanho} erros críticos gerada em {OUTPUT_REPORT}.")
    print("Ação recomendada pelo PM-Engineer: Abra este CSV no Excel e encontre o padrão humano das falhas.")

if __name__ == "__main__":
    generate_error_analysis_sample()
