"""
SVM Classifier Module
Implements SVM-based classification for SSVEP responses
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import joblib
import yaml
import os
import time

class SSVEPClassifier:
    """SVM-based classifier for SSVEP detection"""
    
    def __init__(self, config_path="../../config/settings.yaml"):
        """Initialize SSVEP classifier
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self._load_config(config_path)
        
        # SVM parameters
        self.kernel = self.config['ml']['svm']['kernel']
        self.C = self.config['ml']['svm']['C']
        self.gamma = self.config['ml']['svm']['gamma']
        
        # Model components
        self.svm_model = None
        self.scaler = None
        self.is_trained = False
        
        # Training data storage
        self.training_features = []
        self.training_labels = []
        
        # Performance tracking
        self.training_accuracy = 0.0
        self.cv_scores = []
        
        # Model file paths
        self.model_dir = "models"
        self.model_path = os.path.join(self.model_dir, "ssvep_svm_model.pkl")
        self.scaler_path = os.path.join(self.model_dir, "ssvep_scaler.pkl")
        
        # Create models directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize model with default parameters
        self._initialize_model()
    
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        try:
            # Try relative to this file first
            current_dir = os.path.dirname(os.path.abspath(__file__))
            full_path = os.path.join(current_dir, config_path)
            
            if not os.path.exists(full_path):
                # Try relative to current working directory
                full_path = config_path
            
            with open(full_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load config file {config_path}: {e}")
            # Return default configuration
            return {
                'ml': {
                    'svm': {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'},
                    'training': {'min_samples_per_class': 10, 'cross_validation_folds': 5}
                }
            }
    
    def _initialize_model(self):
        """Initialize the SVM model and scaler"""
        self.svm_model = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            probability=True,  # Enable probability estimates
            random_state=42
        )
        
        self.scaler = StandardScaler()
        print(f"SVM model initialized with kernel={self.kernel}, C={self.C}, gamma={self.gamma}")
    
    def add_training_data(self, features, label):
        """Add training data point
        
        Args:
            features: Feature vector (numpy array)
            label: Class label (0 for freq1, 1 for freq2)
        """
        self.training_features.append(features)
        self.training_labels.append(label)
        print(f"Added training sample with label {label}. Total samples: {len(self.training_labels)}")
    
    def add_training_batch(self, features_batch, labels_batch):
        """Add a batch of training data
        
        Args:
            features_batch: Array of feature vectors (samples x features)
            labels_batch: Array of class labels
        """
        for features, label in zip(features_batch, labels_batch):
            self.add_training_data(features, label)
    
    def clear_training_data(self):
        """Clear all training data"""
        self.training_features = []
        self.training_labels = []
        self.is_trained = False
        print("Training data cleared")
    
    def can_train(self):
        """Check if there's enough data to train the model
        
        Returns:
            bool: True if training is possible
        """
        if len(self.training_labels) < 2:
            return False
        
        # Check if we have samples from both classes
        unique_labels = set(self.training_labels)
        min_samples = self.config['ml']['training']['min_samples_per_class']
        
        for label in [0, 1]:
            if self.training_labels.count(label) < min_samples:
                return False
        
        return True
    
    def train(self, optimize_hyperparameters=False):
        """Train the SVM model
        
        Args:
            optimize_hyperparameters: Whether to perform hyperparameter optimization
            
        Returns:
            bool: True if training was successful
        """
        if not self.can_train():
            print("Not enough training data. Need at least {min_samples_per_class} samples per class.")
            return False
        
        try:
            # Convert training data to numpy arrays
            X = np.array(self.training_features)
            y = np.array(self.training_labels)
            
            print(f"Training SVM with {len(X)} samples and {X.shape[1]} features")
            
            # Standardize features
            X_scaled = self.scaler.fit_transform(X)
            
            # Hyperparameter optimization
            if optimize_hyperparameters:
                print("Optimizing hyperparameters...")
                self._optimize_hyperparameters(X_scaled, y)
            
            # Train the model
            self.svm_model.fit(X_scaled, y)
            
            # Calculate training accuracy
            y_pred = self.svm_model.predict(X_scaled)
            self.training_accuracy = accuracy_score(y, y_pred)
            
            # Perform cross-validation
            cv_folds = self.config['ml']['training']['cross_validation_folds']
            self.cv_scores = cross_val_score(self.svm_model, X_scaled, y, cv=cv_folds)
            
            self.is_trained = True
            
            print(f"Training completed:")
            print(f"  Training accuracy: {self.training_accuracy:.3f}")
            print(f"  Cross-validation scores: {self.cv_scores}")
            print(f"  Mean CV score: {np.mean(self.cv_scores):.3f} (+/- {np.std(self.cv_scores) * 2:.3f})")
            
            return True
            
        except Exception as e:
            print(f"Error during training: {e}")
            return False
    
    def _optimize_hyperparameters(self, X, y):
        """Optimize SVM hyperparameters using GridSearchCV
        
        Args:
            X: Training features
            y: Training labels
        """
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'kernel': ['rbf', 'poly', 'sigmoid']
        }
        
        grid_search = GridSearchCV(
            SVC(probability=True, random_state=42),
            param_grid,
            cv=3,  # Reduced CV folds for speed
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        # Update model with best parameters
        self.svm_model = grid_search.best_estimator_
        self.C = grid_search.best_params_['C']
        self.gamma = grid_search.best_params_['gamma']
        self.kernel = grid_search.best_params_['kernel']
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
    
    def predict(self, features):
        """Make a prediction on new features
        
        Args:
            features: Feature vector (numpy array)
            
        Returns:
            tuple: (prediction, confidence) or (None, 0) if not trained
        """
        if not self.is_trained or self.svm_model is None:
            return None, 0.0
        
        try:
            # Ensure features is 2D
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.svm_model.predict(features_scaled)[0]
            
            # Get prediction probabilities
            probabilities = self.svm_model.predict_proba(features_scaled)[0]
            confidence = np.max(probabilities)
            
            return prediction, confidence
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None, 0.0
    
    def predict_batch(self, features_batch):
        """Make predictions on a batch of features
        
        Args:
            features_batch: Array of feature vectors (samples x features)
            
        Returns:
            tuple: (predictions, confidences) or (None, None) if not trained
        """
        if not self.is_trained or self.svm_model is None:
            return None, None
        
        try:
            # Scale features
            features_scaled = self.scaler.transform(features_batch)
            
            # Make predictions
            predictions = self.svm_model.predict(features_scaled)
            
            # Get prediction probabilities
            probabilities = self.svm_model.predict_proba(features_scaled)
            confidences = np.max(probabilities, axis=1)
            
            return predictions, confidences
            
        except Exception as e:
            print(f"Error during batch prediction: {e}")
            return None, None
    
    def evaluate(self, test_features, test_labels):
        """Evaluate the model on test data
        
        Args:
            test_features: Test feature vectors
            test_labels: True test labels
            
        Returns:
            dict: Evaluation metrics
        """
        if not self.is_trained:
            return {"error": "Model not trained"}
        
        try:
            # Make predictions
            predictions, confidences = self.predict_batch(test_features)
            
            if predictions is None:
                return {"error": "Prediction failed"}
            
            # Calculate metrics
            accuracy = accuracy_score(test_labels, predictions)
            report = classification_report(test_labels, predictions, output_dict=True)
            
            results = {
                "accuracy": accuracy,
                "classification_report": report,
                "mean_confidence": np.mean(confidences),
                "num_samples": len(test_labels)
            }
            
            print(f"Evaluation results:")
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  Mean confidence: {np.mean(confidences):.3f}")
            
            return results
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return {"error": str(e)}
    
    def save_model(self, model_path=None, scaler_path=None):
        """Save the trained model and scaler
        
        Args:
            model_path: Path to save the model (optional)
            scaler_path: Path to save the scaler (optional)
            
        Returns:
            bool: True if successful
        """
        if not self.is_trained:
            print("Cannot save: model not trained")
            return False
        
        try:
            model_file = model_path or self.model_path
            scaler_file = scaler_path or self.scaler_path
            
            # Save model
            joblib.dump(self.svm_model, model_file)
            
            # Save scaler
            joblib.dump(self.scaler, scaler_file)
            
            # Save metadata
            metadata = {
                'training_accuracy': self.training_accuracy,
                'cv_scores': self.cv_scores.tolist() if len(self.cv_scores) > 0 else [],
                'kernel': self.kernel,
                'C': self.C,
                'gamma': self.gamma,
                'num_training_samples': len(self.training_labels),
                'timestamp': time.time()
            }
            
            metadata_file = model_file.replace('.pkl', '_metadata.pkl')
            joblib.dump(metadata, metadata_file)
            
            print(f"Model saved to {model_file}")
            print(f"Scaler saved to {scaler_file}")
            print(f"Metadata saved to {metadata_file}")
            
            return True
            
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, model_path=None, scaler_path=None):
        """Load a trained model and scaler
        
        Args:
            model_path: Path to load the model from (optional)
            scaler_path: Path to load the scaler from (optional)
            
        Returns:
            bool: True if successful
        """
        try:
            model_file = model_path or self.model_path
            scaler_file = scaler_path or self.scaler_path
            
            if not os.path.exists(model_file) or not os.path.exists(scaler_file):
                print(f"Model files not found: {model_file}, {scaler_file}")
                return False
            
            # Load model
            self.svm_model = joblib.load(model_file)
            
            # Load scaler
            self.scaler = joblib.load(scaler_file)
            
            # Load metadata if available
            metadata_file = model_file.replace('.pkl', '_metadata.pkl')
            if os.path.exists(metadata_file):
                metadata = joblib.load(metadata_file)
                self.training_accuracy = metadata.get('training_accuracy', 0.0)
                self.cv_scores = np.array(metadata.get('cv_scores', []))
                self.kernel = metadata.get('kernel', 'rbf')
                self.C = metadata.get('C', 1.0)
                self.gamma = metadata.get('gamma', 'scale')
                
                print(f"Model loaded with:")
                print(f"  Training accuracy: {self.training_accuracy:.3f}")
                print(f"  Training samples: {metadata.get('num_training_samples', 'unknown')}")
                print(f"  Parameters: kernel={self.kernel}, C={self.C}, gamma={self.gamma}")
            
            self.is_trained = True
            print(f"Model loaded from {model_file}")
            print(f"Scaler loaded from {scaler_file}")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def get_training_stats(self):
        """Get training statistics
        
        Returns:
            dict: Training statistics
        """
        return {
            'is_trained': self.is_trained,
            'num_training_samples': len(self.training_labels),
            'training_accuracy': self.training_accuracy,
            'cv_scores': self.cv_scores.tolist() if len(self.cv_scores) > 0 else [],
            'mean_cv_score': np.mean(self.cv_scores) if len(self.cv_scores) > 0 else 0.0,
            'model_parameters': {
                'kernel': self.kernel,
                'C': self.C,
                'gamma': self.gamma
            }
        }
    
    def reset_model(self):
        """Reset the model to untrained state"""
        self._initialize_model()
        self.is_trained = False
        self.training_accuracy = 0.0
        self.cv_scores = []
        print("Model reset to untrained state")
    
    def get_feature_importance(self):
        """Get feature importance (for linear kernels only)
        
        Returns:
            numpy.ndarray or None: Feature importance weights
        """
        if not self.is_trained or self.kernel != 'linear':
            print("Feature importance only available for trained linear SVM models")
            return None
        
        try:
            # For linear SVM, feature importance is the absolute value of coefficients
            importance = np.abs(self.svm_model.coef_[0])
            return importance / np.sum(importance)  # Normalize
            
        except Exception as e:
            print(f"Error calculating feature importance: {e}")
            return None 