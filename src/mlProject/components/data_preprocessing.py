import pandas as pd
import numpy as np
from mlProject import logger
from pathlib import Path
from mlProject.entity.config_entity import DataPreprocessingConfig
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

class DataPreprocessing:
    def __init__(self, config: DataPreprocessing):
        self.config = config
    
    def _validation_check(self):
        logger.info("\nCheck the Status File!")
        try:
            validation_status = None
            with open(self.config.data_validation_status_file,'r') as f:
                file= f.readline()
            if 'Validation Status: True' in file:
                validation_status = True
            else:
                validation_status = False
            return validation_status
        except Exception as e:
            raise e

    def _load_data(self,validation_status):
        if validation_status == True:
            df = pd.read_csv(self.config.source_data)
            logger.info(f"\nData is Successfully loaded and the initial shape is {df.shape}")
            logger.info(f"\nEDA:\n{df.isnull().sum()}\n{df.info()}\n{df.describe()}")
        else:
            raise ValueError
        return df
    
    def _creating_features(self,df):
        logger.info(f"\nAdding New Features")
        beta_rad = np.radians(df['Slope Angle (°)'])
        phi_rad   = np.radians(df['Internal Friction Angle (°)'])
        logger.info(f"\nAdding beta_rad: {beta_rad}\nphi_rad: {phi_rad}")

        df['Stability_Number'] = df['Cohesion (kPa)'] / (df['Unit Weight (kN/m³)'] * df['Slope Height (m)'])
        df['tan_phi']          = np.tan(phi_rad)
        df['Effective_Friction'] = (1 - df['Pore Water Pressure Ratio']) * df['tan_phi']
        df['Height_x_sinbeta'] = df['Slope Height (m)'] * np.sin(beta_rad)
        df['c_over_gamma']     = df['Cohesion (kPa)'] / df['Unit Weight (kN/m³)']          
        df['phi_x_ru']         = df['Internal Friction Angle (°)'] * df['Pore Water Pressure Ratio']
        logger.info(f"New features added. Shape after feature engineering: {df.shape}")

        return df

    def _create_target(self, df):
        """Create binary target: 1 if Factor of Safety >= 1.5, else 0"""
        logger.info("\nCreating binary target (Safety_Status)")
        df['Safety_Status'] = (df['Factor of Safety (FS)'] >= 1.5).astype(int)
        stable_count = df['Safety_Status'].sum()
        unstable_count = len(df) - stable_count
        logger.info(f"Target distribution: Stable (1): {stable_count}, Unstable (0): {unstable_count}")
        return df
    
    def _handle_missing_values(self, df):
        """Check for missing values and handle if any (none in this dataset)"""
        logger.info("\nChecking for missing values")
        missing = df.isnull().sum()
        if missing.any():
            logger.warning(f"Missing values found:\n{missing[missing > 0]}")
            # Simple imputation: drop rows with missing target, fill numeric with median, categorical with mode
            # For simplicity, we drop rows with any missing value in this project
            initial_shape = df.shape
            df = df.dropna()
            logger.info(f"Dropped rows with missing values. New shape: {df.shape} (dropped {initial_shape[0] - df.shape[0]} rows)")
        else:
            logger.info("No missing values found.")
        return df
    
    def _encode_categorical(self, df):
        """One‑hot encode 'Reinforcement Numeric' and return encoded dataframe + encoder object"""
        logger.info("\nEncoding categorical feature: Reinforcement Numeric")
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        encoded = encoder.fit_transform(df[['Reinforcement Numeric']])
        feature_names = encoder.get_feature_names_out(['Reinforcement Numeric']).tolist()
        encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)
        # Drop original column and concatenate
        df = df.drop(['Reinforcement Type', 'Reinforcement Numeric'], axis=1)
        df = pd.concat([df, encoded_df], axis=1)
        logger.info(f"One‑hot encoding complete. Added columns: {feature_names}")

        return df, encoder
    
    def _scale_numerical(self, df, fit_scaler=True):
        """Scale numerical features using StandardScaler.
           Returns scaled dataframe and the scaler object."""
        logger.info("\nScaling numerical features")
        # Define numerical columns (exclude target and already encoded categorical)
        exclude_cols = ['Safety_Status', 'Factor of Safety (FS)']
        numerical_cols = df.select_dtypes(include=[np.number]).columns.difference(exclude_cols).tolist()
        scaler = StandardScaler()
        if fit_scaler:
            scaled = scaler.fit_transform(df[numerical_cols])
        else:
            scaled = scaler.transform(df[numerical_cols])
        
        scaled_df = pd.DataFrame(scaled, columns=numerical_cols, index=df.index)
        # Replace original numerical columns with scaled versions
        df = df.drop(columns=numerical_cols)
        df = pd.concat([df, scaled_df], axis=1)
        logger.info(f"Scaled {len(numerical_cols)} numerical features.")

        return df, scaler
    
    def _split_data(self, df):
        """Split into train and test sets, stratifying on target."""
        logger.info("\nSplitting data into train/test sets")
        X = df.drop(['Factor of Safety (FS)', 'Safety_Status'], axis=1)
        X.to_csv(self.config.X,index=False)
        logger.info(f"\nX data is completely saved at {self.config.X}")
        y = df['Safety_Status']
        y.to_csv(self.config.y,index=False)
        logger.info(f"\ny data is completely saved at {self.config.y}")
        logger.info(f"\nX shape is: {X.shape}\ny shape is: {y.shape}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info(f"Train size: {X_train.shape[0]} rows, Test size: {X_test.shape[0]} rows")
        logger.info(f"Train class distribution:\n{y_train.value_counts()}")
        logger.info(f"Test class distribution:\n{y_test.value_counts()}")

        return X_train, X_test, y_train, y_test
    
    def _save_preprocessed_data(self, X_train, X_test, y_train, y_test):
        """Save train/test datasets to CSV files."""
        logger.info("\nSaving preprocessed datasets")
        
        
        X_train.to_csv(self.config.train_test_split/'X_train.csv', index=False)
        X_test.to_csv(self.config.train_test_split/'X_test.csv', index=False)
        y_train.to_csv(self.config.train_test_split/'y_train.csv', index=False)
        y_test.to_csv(self.config.train_test_split/'y_test.csv', index=False)
        logger.info(f"Saved X_train to {self.config.train_test_split}")
        logger.info(f"Saved X_test to {self.config.train_test_split}")
        logger.info(f"Saved y_train to {self.config.train_test_split}")
        logger.info(f"Saved y_test to {self.config.train_test_split}")
    
    def _save_preprocessor(self, encoder, scaler):
        """Save the fitted encoder and scaler for later use in prediction."""
        logger.info("\nSaving preprocessor objects")
        os.makedirs(self.config.root_dir, exist_ok=True)
        joblib.dump(encoder, self.config.train_test_split/'encoder.joblib')
        joblib.dump(scaler, self.config.train_test_split/'scaler.joblib')
        logger.info(f"Encoder saved to {self.config.train_test_split/'encoder.joblib'}")
        logger.info(f"Scaler saved to {self.config.train_test_split/'scaler.joblib'}")
    
    def initiate_data_preprocessing(self):
        """
        Orchestrates the entire data preprocessing pipeline.
        """
        logger.info("\n========== STARTING DATA PREPROCESSING STAGE ==========")
        # 1. Validation check
        validation_status = self._validation_check()
        # 2. Load data
        df = self._load_data(validation_status)
        # 3. Create new features
        df = self._creating_features(df)
        # 4. Create binary target
        df = self._create_target(df)
        # 5. Handle missing values (if any)
        df = self._handle_missing_values(df)
        # 6. Encode categorical variable
        df, encoder = self._encode_categorical(df)
        # 7. Scale numerical features (fit on whole dataset before splitting to avoid data leakage? 
        #    Better to fit on training set only, but for simplicity we split first then scale.
        #    We'll split first, then scale using scaler fitted on training set.
        #    So we separate split and scaling.
        
        # 8. Split data (before scaling to avoid data leakage)
        X_train, X_test, y_train, y_test = self._split_data(df)
        
        # 9. Scale numerical features (fit on train, transform train+test)
        #    Re-identify numerical columns from the split data (target already removed)
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
        logger.info(f"Scaled {len(numerical_cols)} numerical features on train/test sets.")
        
        # 10. Save preprocessed data
        self._save_preprocessed_data(X_train_scaled, X_test_scaled, y_train, y_test)
        # 11. Save preprocessor objects (encoder and scaler)
        self._save_preprocessor(encoder, scaler)
        
        logger.info("\n========== DATA PREPROCESSING STAGE COMPLETED ==========")
