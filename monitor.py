import pandas as pd
import json
import os
from utils import criar_features

def check_drift(new_data_path, metadata_path='models/v1/metadata.json'):
    print("Iniciando Verificacao de Data Drift...")
    
    if not os.path.exists(new_data_path):
        print(f"Erro: Arquivo {new_data_path} nao encontrado.")
        return
        
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    new_df = pd.read_csv(new_data_path)
    new_df = criar_features(new_df)
    
    # Exemplo simples: Comparar media de 'delivery_delay_days'
    # Em um sistema real, usariamos Testes de Kolmogorov-Smirnov ou similar.
    # Aqui faremos uma verificacao de threshold operacional.
    
    train_features = metadata['features']
    # Simulando medias de treino para exemplo (num sistema real isso estaria no metadata)
    # Aqui apenas demonstraremos a estrutura do monitor
    
    current_avg_delay = new_df['delivery_delay_days'].mean()
    print(f"Media atual de atraso: {current_avg_delay:.2f} dias")
    
    if current_avg_delay > 5.0: # Threshold arbitrario para exemplo
        print("ALERTA: Possivel Drift detectado na performance logistica!")
        print("Recomendado: Re-treinamento do modelo.")
    else:
        print("Performance logistica dentro dos limites esperados.")

if __name__ == "__main__":
    # Usando o mesmo arquivo para teste
    check_drift('data/desafio_nps_fase_1.csv')
