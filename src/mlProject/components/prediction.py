import pandas as pd
import joblib
import os
from pathlib import Path
from mlProject import logger
from mlProject.entity.config_entity import PredictionConfig

class PredictionPipeline:
    def __init__(self, config: PredictionConfig):
        self.config = config

    def _load_artifacts(self):
        """Load the trained model and scaler."""
        logger.info("Loading model and scaler")
        self.model = joblib.load(self.config.model_path)
        self.scaler = joblib.load(self.config.scaler_path)
        logger.info("Model and scaler loaded successfully")

    def _load_data(self):
        """Load preprocessed features (X) and optional true labels (y)."""
        logger.info(f"Loading features from {self.config.X_path}")
        X = pd.read_csv(self.config.X_path)
        logger.info(f"Features shape: {X.shape}")

        y = None
        if self.config.y_path and Path(self.config.y_path).exists():
            logger.info(f"Loading true labels from {self.config.y_path}")
            y = pd.read_csv(self.config.y_path).squeeze("columns")
            logger.info(f"Labels shape: {len(y)}")
        else:
            logger.info("No true labels provided or file not found.")
        return X, y

    def _scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale features using the fitted scaler."""
        logger.info("Applying standard scaling")
        # Ensure column order matches the scaler's expected order
        if hasattr(self.scaler, 'feature_names_in_'):
            expected_cols = list(self.scaler.feature_names_in_)
            missing = set(expected_cols) - set(X.columns)
            if missing:
                raise ValueError(f"Missing columns required by scaler: {missing}")
            X = X[expected_cols]  # reorder columns
        else:
            logger.warning("Scaler does not have feature_names_in_. Assuming columns are in correct order.")
        scaled = self.scaler.transform(X)
        return pd.DataFrame(scaled, columns=X.columns, index=X.index)

    def run(self) -> pd.DataFrame:
        """Execute the full prediction pipeline and save results."""
        logger.info("\n========== STARTING PREDICTION STAGE ==========")

        # 1. Load artifacts
        self._load_artifacts()

        # 2. Load preprocessed data
        X, y_true = self._load_data()

        # 3. Scale features
        X_scaled = self._scale_features(X)

        # 4. Generate predictions
        logger.info("Generating predictions")
        preds = self.model.predict(X_scaled)
        probs = self.model.predict_proba(X_scaled)[:, 1]

        # 5. Build output DataFrame
        output = X.copy()  # keep original (unscaled) features for reference
        output['Predicted_Safety_Status'] = preds
        output['Probability_Stable'] = probs
        output['Safety_Assessment'] = output['Predicted_Safety_Status'].map({1: 'STABLE', 0: 'UNSTABLE'})

        if y_true is not None:
            output['True_Label'] = y_true.values
            output['Correct'] = (output['Predicted_Safety_Status'] == output['True_Label'])

        # 6. Save to CSV
        os.makedirs(self.config.root_dir, exist_ok=True)
        output.to_csv(self.config.output_path, index=False)
        logger.info(f"Predictions saved to {self.config.output_path}")
        logger.info(f"Output shape: {output.shape}")

        logger.info("\n========== PREDICTION STAGE COMPLETED ==========")
        return output