import joblib
import pandas as pd
from pathlib import Path
import sys

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.quantile_regressor import QuantileRegressor

def main():
    print("Extracting Feature Importance...")
    
    model = QuantileRegressor()
    try:
        model.load_models('models/quantile_forecast')
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    importance_df = model.get_feature_importance()
    
    if importance_df is not None:
        print("\nTop 20 Most Influential Features (Mean Gain):")
        print(importance_df.head(20))
        
        # Save to results
        Path('results').mkdir(exist_ok=True)
        importance_df.to_csv('results/feature_importance.csv')
        print("\nFull feature importance saved to results/feature_importance.csv")
    else:
        print("No feature importance found.")

if __name__ == "__main__":
    main()
