import pandas as pd
import numpy as np
import json
import os
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from mlProject import logger
from mlProject.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def _load_data(self):
        """Load train/test features and targets."""
        logger.info("Loading preprocessed data for training")
        X_train = pd.read_csv(self.config.train_data_path)
        X_test = pd.read_csv(self.config.test_data_path)
        y_train = pd.read_csv(self.config.train_target_path).squeeze("columns")
        y_test = pd.read_csv(self.config.test_target_path).squeeze("columns")
        logger.info(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        logger.info(f"Train class distribution:\n{y_train.value_counts()}")
        logger.info(f"Test class distribution:\n{y_test.value_counts()}")
        return X_train, X_test, y_train, y_test

    def _train_random_forest(self, X_train, y_train):
        """Train Random Forest with class_weight='balanced'."""
        logger.info("Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            random_state=self.config.random_state,
            class_weight=self.config.class_weight,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        logger.info("Random Forest training completed")
        return rf

    def _train_xgboost(self, X_train, y_train):
        """Train XGBoost with scale_pos_weight computed from data."""
        logger.info("Training XGBoost...")
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        scale_pos_weight = neg / pos
        logger.info(f"Computed scale_pos_weight = {scale_pos_weight:.4f}")

        xgb = XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            random_state=self.config.random_state,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        )
        xgb.fit(X_train, y_train)
        logger.info("XGBoost training completed")
        return xgb

    def _evaluate_model(self, model, X_test, y_test, model_name):
        """Compute and log evaluation metrics."""
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = {
            'model_name': model_name,
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1': float(f1_score(y_test, y_pred, zero_division=0)),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        if y_proba is not None:
            metrics['roc_auc'] = float(roc_auc_score(y_test, y_proba))

        logger.info(f"{model_name} - Acc: {metrics['accuracy']:.4f}, Prec: {metrics['precision']:.4f}, "
                    f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}, "
                    f"AUC: {metrics.get('roc_auc', 'N/A')}")
        return metrics

    def _save_metrics(self, metrics_dict):
        """Save metrics to JSON."""
        os.makedirs(self.config.root_dir, exist_ok=True)
        with open(self.config.metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=4)
        logger.info(f"Metrics saved to {self.config.metrics_path}")

    def initiate_model_trainer(self):
        """Orchestrate the full training pipeline."""
        logger.info("\n========== STARTING MODEL TRAINER STAGE ==========")

        # 1. Load data
        X_train, X_test, y_train, y_test = self._load_data()

        # 2. Train both models
        rf_model = self._train_random_forest(X_train, y_train)
        xgb_model = self._train_xgboost(X_train, y_train)

        # 3. Evaluate both
        metrics = {}
        metrics['RandomForest'] = self._evaluate_model(rf_model, X_test, y_test, 'RandomForest')
        metrics['XGBoost'] = self._evaluate_model(xgb_model, X_test, y_test, 'XGBoost')

        # 4. Select best model based on F1 (minority class)
        best_model_name = max(metrics, key=lambda x: metrics[x]['f1'])
        best_model = rf_model if best_model_name == 'RandomForest' else xgb_model
        logger.info(f"Best model: {best_model_name} (F1 = {metrics[best_model_name]['f1']:.4f})")

        # 5. Save models
        os.makedirs(self.config.root_dir, exist_ok=True)
        joblib.dump(rf_model, self.config.rf_model_path)
        joblib.dump(xgb_model, self.config.xgb_model_path)
        joblib.dump(best_model, self.config.model_path)
        logger.info(f"RandomForest saved to {self.config.rf_model_path}")
        logger.info(f"XGBoost saved to {self.config.xgb_model_path}")
        logger.info(f"Best model saved to {self.config.model_path}")

        # 6. Save metrics
        self._save_metrics(metrics)

        logger.info("\n========== MODEL TRAINER STAGE COMPLETED ==========")
        return metrics, best_model_name