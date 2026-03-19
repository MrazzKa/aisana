import numpy as np
import pandas as pd


def explain_prediction(model, scaler, feature_names: list, category_mappings: dict, input_data: dict) -> list:
    """Explain a single prediction using feature importance and deviation from mean.

    For each feature, the contribution is estimated as:
        contribution = feature_importance * (scaled_value - 0)
    Since the scaler centres data around 0, the scaled value itself represents
    deviation from the training mean in units of standard deviation.

    Returns a list of dicts sorted by absolute contribution:
        [{feature, value, contribution}, ...]
    """
    # Build a single-row DataFrame
    row = {}
    for feat in feature_names:
        if feat in input_data:
            row[feat] = input_data[feat]
        else:
            row[feat] = 0

    df = pd.DataFrame([row], columns=feature_names)

    # Apply category mappings
    for col, mapping in category_mappings.items():
        if col in df.columns:
            df[col] = df[col].astype(str).map(mapping).fillna(-1)

    df = df.fillna(0)

    # Scale
    scaled = scaler.transform(df)

    # Get feature importances
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        # Uniform importance fallback
        importances = np.ones(len(feature_names)) / len(feature_names)

    # Compute contributions: importance * scaled_value (deviation from mean)
    contributions = importances * scaled[0]

    results = []
    for i, feat in enumerate(feature_names):
        original_value = input_data.get(feat, 0)
        results.append({
            "feature": feat,
            "value": original_value,
            "contribution": round(float(contributions[i]), 6),
        })

    # Sort by absolute contribution descending
    results.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return results
