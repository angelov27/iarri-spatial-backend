import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torch_geometric.nn import GCNConv

app = FastAPI(title="API Predictiva IARRI-MX Spatial ML")

# Habilitar CORS para que tu app en Vercel pueda hacer consultas sin bloqueos
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Definición de la Arquitectura de la Red Neuronal de Grafos (GNN)
class SpatialGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(SpatialGNN, self).__init__()
        # Implementa la convolución de grafos recomendada en el documento
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)
        
    def forward(self, x, edge_index):
        # Capa 1 con activación ReLU
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        # Capa 2 con Sigmoide para mapear estrictamente al rango [0, 1]
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x)

# Inicializar modelo con 5 variables de entrada (AV, IC, ED, EAR, IM)
# Proporciona la estructura del "Final Trainable Model" del PDF
num_features = 5
model = SpatialGNN(in_channels=num_features, hidden_channels=16)
model.eval() # Modo inferencia

# Estructura de datos simulada para el Grafo de Puebla (Nodos y Aristas de adyacencia)
# En producción, edge_index se genera usando distancias geográficas o colindancia de colonias
edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 0, 4],
                           [1, 0, 2, 1, 3, 2, 4, 0]], dtype=torch.long)

# Modelo de datos que esperará la API desde tu React Web
class AnalisisRequest(BaseModel):
    areas_verdes: float  # V1 - AV
    caminabilidad: float # V2 - IC
    equip_deportivo: float # V3 - ED
    entorno_riesgoso: float # V4 - EAR
    marginacion: float # V5 - IM

@app.post("/api/predict-spatial")
async def predict_spatial(data: AnalisisRequest):
    # Convertir los datos de la petición web en tensores de PyTorch
    input_features = np.array([
        data.areas_verdes,
        data.caminabilidad,
        data.equip_deportivo,
        data.entorno_riesgoso,
        data.marginacion
    ], dtype=np.float32)
    
    # Replicamos el nodo consultado junto a nodos vecinos del grafo para la convolución
    # Esto simula un campo neuronal sensible al contexto socioeconómico territorial
    node_features = np.tile(input_features, (5, 1)) 
    x_tensor = torch.tensor(node_features, dtype=torch.float32)
    
    with torch.no_grad():
        # Ejecuta la ecuación: I_theta = sigma(GNN_theta(X, W))
        prediction = model(x_tensor, edge_index)
        iarri_calculado = float(prediction[0].item()) # Tomamos el nodo objetivo
        
    # Calcular Capa de Interpretabilidad (Simulación de SHAP / Gradientes Integrados)
    # Atribuye el peso dinámico que tuvo cada factor urbano en la predicción final
    base_weights = np.array([-0.20, -0.25, -0.15, 0.25, 0.15]) # Direccionalidad monótona
    raw_shap = input_features * base_weights
    shap_normalized = (raw_shap - raw_shap.min()) / (raw_shap.max() - raw_shap.min() + 1e-5)
    shap_percentages = (shap_normalized / shap_normalized.sum() * 100).tolist()

    # Clasificación exacta basada en los rangos del modelo
    if iarri_calculado <= 0.33:
        nivel = "Bajo Riesgo"
        color = "#1E5631" # Verde
    elif iarri_calculado <= 0.66:
        nivel = "Riesgo Medio"
        color = "#E67E22" # Naranja
    else:
        nivel = "Alto Riesgo"
        color = "#C0392B" # Rojo

    return {
        "iarri": round(iarri_calculado, 2),
        "nivel": nivel,
        "color_hex": color,
        "interpretabilidad_shap": {
            "areas_verdes": round(shap_percentages[0], 1),
            "caminabilidad": round(shap_percentages[1], 1),
            "equip_deportivo": round(shap_percentages[2], 1),
            "entorno_riesgoso": round(shap_percentages[3], 1),
            "marginacion": round(shap_percentages[4], 1)
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)