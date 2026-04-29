import pandas as pd
import numpy as np

def criar_features(df_input):
    """
    Função de Feature Engineering para o projeto NPS.
    Centraliza a lógica de criação de variáveis para garantir paridade entre Treino e Produção.
    """
    df = df_input.copy()
    
    # Feature 1: Ratio de atraso em relação ao prazo (Atraso Relativo)
    # Por que: 3 dias de atraso em 5 dias de prazo é pior que em 20 dias.
    df['ratio_atraso_entrega'] = df['delivery_delay_days'] / (df['delivery_time_days'] + 1)

    # Feature 2: Custo total por item (incluindo frete)
    df['custo_por_item'] = (df['order_value'] + df['freight_value']) / df['items_quantity']

    # Feature 3: Indice de intensidade do problema com suporte
    df['intensidade_problema'] = (df['customer_service_contacts'] *
                                   df['resolution_time_days'] *
                                   (df['complaints_count'] + 1))

    # Feature 4: Flag binaria de entrega pontual
    df['entrega_no_prazo'] = (df['delivery_delay_days'] == 0).astype(int)

    # Feature 5: Score composto de experiencia logistica
    df['score_logistica'] = (
        - df['delivery_delay_days'] * 2
        - df['delivery_attempts']
        + df['entrega_no_prazo'] * 5
    )

    # Feature 6: Flag de cliente de longa data
    df['cliente_longa_data'] = (df['customer_tenure_months'] > 60).astype(int)

    # Feature 7: Percentual de desconto relativo ao valor do pedido
    df['pct_desconto'] = df['discount_value'] / (df['order_value'] + 1) * 100

    return df

FEATURES_MODELO = [
    'customer_age', 'customer_tenure_months', 'order_value', 'items_quantity',
    'discount_value', 'payment_installments', 'delivery_time_days',
    'delivery_delay_days', 'freight_value', 'delivery_attempts',
    'customer_service_contacts', 'resolution_time_days', 'complaints_count',
    'ratio_atraso_entrega', 'custo_por_item', 'intensidade_problema',
    'entrega_no_prazo', 'score_logistica', 'cliente_longa_data', 'pct_desconto'
]
