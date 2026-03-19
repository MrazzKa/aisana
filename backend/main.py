import os
import io
import json
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional

from ml.trainer import train_model, SUPPORTED_MODEL_NAMES
from ml.predictor import predict, predict_single, predict_with_threshold
from stats import record_training, record_prediction, get_stats
from rules import get_rules, add_rule, update_rule, delete_rule, apply_rules

app = FastAPI(title="AISana ML API")

# Setup CORS to allow Next.js app to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
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

    # Load stats for metrics lookup
    stats = get_stats()
    # Build a map: model_name -> latest training metrics
    latest_metrics = {}
    for activity in stats.get("recent_activity", []):
        if activity["type"] == "training":
            name = activity["model_name"]
            if name not in latest_metrics:
                latest_metrics[name] = activity.get("detail", "")

    # Also build full metrics from stats file
    stats_data = _load_stats_raw()
    model_metrics_map = {}
    for t in stats_data.get("trainings", []):
        model_metrics_map[t["model_name"]] = t.get("metrics", {})

    trained_models = []
    for f in os.listdir(models_dir):
        if f.endswith("_model.pkl"):
            model_name = f.replace("_model.pkl", "")

            features = []
            features_path = os.path.join(models_dir, f"{model_name}_features.json")
            if os.path.exists(features_path):
                with open(features_path, "r") as json_file:
                    features = json.load(json_file)

            trained_models.append({
                "name": model_name,
                "features": features,
                "metrics": model_metrics_map.get(model_name, {}),
            })

    return trained_models


def _load_stats_raw() -> dict:
    """Helper to load raw stats JSON for metrics lookup."""
    stats_file = "stats.json"
    if not os.path.exists(stats_file):
        return {"trainings": [], "predictions": []}
    try:
        with open(stats_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {"trainings": [], "predictions": []}


@app.get("/api/stats")
def api_stats():
    return get_stats()


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

        # Record training in stats
        record_training(model_name, results.get("test_metrics", {}))

        return results

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict")
async def api_predict(
    model_name: str = Form(...),
    file: UploadFile = File(...),
    threshold: Optional[str] = Form(None),
):
    try:
        contents = await file.read()
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_excel(io.BytesIO(contents))

        thresh = 0.5
        if threshold is not None:
            try:
                thresh = float(threshold)
            except ValueError:
                pass

        results = predict_with_threshold(df, model_name, threshold=thresh)

        # Record prediction in stats
        summary = results.get("summary", {})
        record_prediction(model_name, summary.get("total", 0), summary.get("fraud_count", 0))

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict-single")
async def api_predict_single(body: dict = Body(...)):
    """Predict fraud for a single transaction.

    Accepts JSON body: {model_name, data: {field: value, ...}, threshold?}
    """
    try:
        model_name = body.get("model_name")
        if not model_name:
            raise HTTPException(status_code=400, detail="model_name is required")

        data = body.get("data", {})
        if not data:
            raise HTTPException(status_code=400, detail="data is required")

        threshold = float(body.get("threshold", 0.5))
        result = predict_single(data, model_name, threshold=threshold)

        # Also check rules
        rules_result = apply_rules(data)
        result["rules_result"] = rules_result

        # Record in stats
        fraud_count = 1 if result.get("is_fraud", 0) == 1 else 0
        record_prediction(model_name, 1, fraud_count)

        return result

    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/compare")
async def api_compare(
    use_default: str = Form(...),
    file: UploadFile = File(None),
):
    """Train ALL supported models on the same dataset and return comparison."""
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

        comparison = []
        for model_name in SUPPORTED_MODEL_NAMES:
            try:
                # train_model creates a fresh instance via factory each time
                result = train_model(df.copy(), model_name)
                record_training(model_name, result.get("test_metrics", {}))
                comparison.append({
                    "model_name": model_name,
                    "test_metrics": result.get("test_metrics", {}),
                    "train_metrics": result.get("train_metrics", {}),
                    "train_time": result.get("test_metrics", {}).get("train_time", 0),
                    "features_used": result.get("features_used", []),
                })
            except Exception as model_err:
                comparison.append({
                    "model_name": model_name,
                    "error": str(model_err),
                })

        return {"results": comparison}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---- Rules endpoints ----

@app.get("/api/rules")
def api_get_rules():
    return get_rules()


@app.post("/api/rules")
def api_add_rule(rule: dict = Body(...)):
    try:
        return add_rule(rule)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.put("/api/rules/{rule_id}")
def api_update_rule(rule_id: str, rule: dict = Body(...)):
    try:
        return update_rule(rule_id, rule)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/api/rules/{rule_id}")
def api_delete_rule(rule_id: str):
    try:
        delete_rule(rule_id)
        return {"status": "deleted"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/check-rules")
def api_check_rules(transaction: dict = Body(...)):
    """Check a transaction against all enabled rules."""
    return apply_rules(transaction)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
