"""
Utilities to save trained models and metrics
"""

import json
import pickle
import os
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models" / "latest"
METRICS_DIR = BASE_DIR / "metrics"

def save_model(model, model_name):
    """Save trained model to pickle file"""
    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / f"{model_name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"✓ Saved model: {model_path}")
        return True
    except Exception as e:
        print(f"✗ Error saving model {model_name}: {e}")
        return False

def save_metrics(model_name, metrics_dict):
    """Save metrics to JSON file"""
    try:
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        metrics_path = METRICS_DIR / f"{model_name}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2, default=str)
        print(f"✓ Saved metrics: {metrics_path}")
        return True
    except Exception as e:
        print(f"✗ Error saving metrics for {model_name}: {e}")
        return False

def save_feature_columns(model_name, feature_columns):
    """Save feature column names"""
    try:
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        features_path = METRICS_DIR / f"{model_name}_features.json"
        features_dict = {"features": list(feature_columns), "count": len(feature_columns)}
        with open(features_path, 'w') as f:
            json.dump(features_dict, f, indent=2)
        print(f"✓ Saved features for {model_name}")
        return True
    except Exception as e:
        print(f"✗ Error saving features for {model_name}: {e}")
        return False

def aggregate_all_metrics(metrics_by_model):
    """
    Aggregate metrics from all models into one JSON
    metrics_by_model: dict like {
        'lin_base': {...},
        'lin_model2': {...},
        ...
    }
    """
    try:
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        latest_metrics_path = METRICS_DIR / "latest_metrics.json"

        # Merge with existing metrics if present
        existing = {}
        if latest_metrics_path.exists():
            try:
                with open(latest_metrics_path, 'r') as f:
                    existing = json.load(f).get('models', {})
            except Exception:
                existing = {}

        merged_models = {**existing, **metrics_by_model}

        aggregated = {
            'timestamp': datetime.now().isoformat(),
            'models': merged_models
        }

        with open(latest_metrics_path, 'w') as f:
            json.dump(aggregated, f, indent=2, default=str)
        print(f"✓ Saved all metrics: {latest_metrics_path}")
        return True
    except Exception as e:
        print(f"✗ Error saving aggregated metrics: {e}")
        return False

if __name__ == "__main__":
    print("Testing model_saver...")
    # Test functions
    test_model = {"test": "model"}
    test_metrics = {"accuracy": 0.85, "auc": 0.90}
    test_features = ["feature1", "feature2", "feature3"]
    
    save_model(test_model, "test_model")
    save_metrics("test", test_metrics)
    save_feature_columns("test", test_features)
    
    print("model_saver test complete!")
