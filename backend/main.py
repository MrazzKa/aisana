import os
import io
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ml.trainer import train_model
from ml.predictor import predict

app = FastAPI(title="AISana ML API")

# Setup CORS to allow Next.js app to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def healthcheck():
    return {"status": "ok"}

@app.get("/api/models")
def get_models():
    models_dir = "models"
    if not os.path.exists(models_dir):
        return []
    
    trained_models = []
    for f in os.listdir(models_dir):
        if f.endswith("_model.pkl"):
            model_name = f.replace("_model.pkl", "")
            
            features = []
            features_path = os.path.join(models_dir, f"{model_name}_features.json")
            if os.path.exists(features_path):
                import json
                with open(features_path, "r") as json_file:
                    features = json.load(json_file)
                    
            trained_models.append({
                "name": model_name,
                "features": features
            })
            
    return trained_models

@app.post("/api/train")
async def api_train(
    model_name: str = Form(...),
    use_default: str = Form(...),
    file: UploadFile = File(None)
):
    try:
        if use_default.lower() == "true":
            if not os.path.exists("account_data.csv"):
                raise HTTPException(status_code=400, detail="Default dataset not found")
            df = pd.read_csv("account_data.csv")
        else:
            if file is None:
                raise HTTPException(status_code=400, detail="No file provided")
            contents = await file.read()
            if file.filename.endswith('.csv'):
                df = pd.read_csv(io.BytesIO(contents))
            else:
                df = pd.read_excel(io.BytesIO(contents))
                
        if "IsFraud" not in df.columns:
            raise HTTPException(status_code=400, detail="Column 'IsFraud' missing from dataset")
            
        results = train_model(df, model_name)
        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict")
async def api_predict(
    model_name: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        contents = await file.read()
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_excel(io.BytesIO(contents))
            
        results = predict(df, model_name)
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
