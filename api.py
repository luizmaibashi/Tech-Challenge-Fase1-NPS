from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn
from utils import criar_features, FEATURES_MODELO

# Inicialização da App
app = FastAPI(
    title="NPS Predictor API",
    description="API para predição de classe NPS com base em dados operacionais logísticos.",
    version="1.0.0"
)

# Carregamento do Modelo
MODEL_PATH = 'models/v1/pipeline_completo.pkl'
try:
    model = joblib.load(MODEL_PATH)
    print(f"Modelo carregado com sucesso de {MODEL_PATH}")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    model = None

# Esquema de Dados de Entrada
class PedidoInput(BaseModel):
    customer_age: int
    customer_tenure_months: int
    order_value: float
    items_quantity: int
    discount_value: float
    payment_installments: int
    delivery_time_days: int
    delivery_delay_days: int
    freight_value: float
    delivery_attempts: int
    customer_service_contacts: int
    resolution_time_days: int
    complaints_count: int

@app.get("/")
def home():
    return {"status": "online", "model_version": "v1"}

@app.post("/predict")
def predict(pedido: PedidoInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo não disponível no servidor.")
    
    try:
        # 1. Converter entrada para DataFrame
        df_input = pd.DataFrame([pedido.model_dump()])
        
        # 2. Aplicar Feature Engineering
        df_transformed = criar_features(df_input)
        
        # 3. Selecionar apenas as colunas usadas no modelo
        X = df_transformed[FEATURES_MODELO]
        
        # 4. Predição
        prediction = int(model.predict(X)[0])
        probabilities = model.predict_proba(X)[0].tolist()
        
        classes = {0: "Detrator", 1: "Neutro", 2: "Promotor"}
        
        return {
            "prediction": prediction,
            "label": classes[prediction],
            "probabilities": {
                "Detrator": probabilities[0],
                "Neutro": probabilities[1],
                "Promotor": probabilities[2]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
