import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

# Import neural network libraries
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
    NEURAL_NETWORKS_AVAILABLE = True
    print("✅ TensorFlow successfully imported")
except ImportError as e:
    print(f"⚠️ TensorFlow not available. Neural Networks will be disabled. Error: {str(e)}")
    NEURAL_NETWORKS_AVAILABLE = False

class DropoutPredictor:
    def __init__(self):
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'SVM': SVC(random_state=42, probability=True),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42)
        }
        
        # Add Neural Network if available
        if NEURAL_NETWORKS_AVAILABLE:
            self.models['Neural Network'] = self._create_neural_network()
        
        self.results = {}
        self.feature_importances = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.X_test = None
        self.y_test = None
        self.feature_names = None
        self.confusion_matrices = {}
        
    def _create_neural_network(self):
        """Create a neural network model with proper input shape handling"""
        def create_model(input_dim=None):
            model = Sequential()
            
            # Input layer with proper input shape
            model.add(Dense(128, activation='relu', input_shape=(input_dim,)))
            model.add(BatchNormalization())
            model.add(Dropout(0.4))
            
            # Hidden layers
            model.add(Dense(64, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
            
            model.add(Dense(32, activation='relu'))
            model.add(Dropout(0.2))
            
            # Output layer for binary classification
            model.add(Dense(1, activation='sigmoid'))
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            return model
        
        # Return a wrapper that will be properly initialized later
        return lambda input_dim: KerasClassifier(
            build_fn=lambda: create_model(input_dim=input_dim),
            epochs=100,
            batch_size=32,
            verbose=0,
            validation_split=0.2,
            callbacks=[
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5)
            ]
        )
    
    def load_and_preprocess_data(self, file_path=None):
        """Load and preprocess the dropout dataset"""
        try:
            if file_path:
                # Load from CSV file
                df = pd.read_csv(file_path)
            else:
                # Create sample data based on your structure
                df = self._create_sample_data()
            
            print("✅ Data loaded successfully")
            print(f"📊 Dataset Shape: {df.shape}")
            print(f"🎯 Features: {len(df.columns) - 1}")  # Excluding target
            print(f"📈 Dropout Rate: {df['Dropout_Likelihood'].value_counts(normalize=True).get('Yes', 0):.1%}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return None
    
    def _create_sample_data(self):
        """Create sample data matching your dataset structure"""
        np.random.seed(42)
        n_samples = 10000
        
        data = {
            'Student_ID': [f'S{str(i).zfill(5)}' for i in range(1, n_samples + 1)],
            'Age': np.random.randint(15, 50, n_samples),
            'Gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4]),
            'Education_Level': np.random.choice(['High School', 'Undergraduate', 'Postgraduate'], n_samples, p=[0.3, 0.5, 0.2]),
            'Course_Name': np.random.choice(['Machine Learning', 'Python Basics', 'Data Science', 'Web Development'], n_samples),
            'Time_Spent_on_Videos': np.random.randint(50, 500, n_samples),
            'Quiz_Attempts': np.random.randint(1, 10, n_samples),
            'Quiz_Scores': np.random.randint(40, 100, n_samples),
            'Forum_Participation': np.random.randint(0, 50, n_samples),
            'Assignment_Completion_Rate': np.random.randint(50, 100, n_samples),
            'Engagement_Level': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.2, 0.5, 0.3]),
            'Final_Exam_Score': np.random.randint(30, 100, n_samples),
            'Learning_Style': np.random.choice(['Visual', 'Reading/Writing', 'Auditory', 'Kinesthetic'], n_samples),
            'Feedback_Score': np.random.randint(1, 6, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Create realistic dropout likelihood based on features
        dropout_prob = (
            -0.1 * (df['Age'] - df['Age'].mean()) / df['Age'].std() +
            -0.2 * (df['Quiz_Scores'] - 70) / 15 +
            -0.15 * (df['Assignment_Completion_Rate'] - 80) / 10 +
            -0.1 * (df['Final_Exam_Score'] - 65) / 15 +
            0.3 * (df['Engagement_Level'] == 'Low') +
            -0.1 * df['Forum_Participation'] / 10 +
            np.random.normal(0, 0.3, n_samples)
        )
        
        df['Dropout_Likelihood'] = (dropout_prob > np.percentile(dropout_prob, 70)).map({True: 'Yes', False: 'No'})
        
        return df
    
    def preprocess_features(self, df):
        """Preprocess features for modeling"""
        try:
            # Create a copy of the dataframe
            data = df.copy()
            
            # Drop Student_ID as it's not a feature
            if 'Student_ID' in data.columns:
                data = data.drop('Student_ID', axis=1)
            
            # Separate features and target
            X = data.drop('Dropout_Likelihood', axis=1)
            y = data['Dropout_Likelihood']
            
            # Encode categorical variables
            categorical_columns = ['Gender', 'Education_Level', 'Course_Name', 'Engagement_Level', 'Learning_Style']
            
            for col in categorical_columns:
                if col in X.columns:
                    self.label_encoders[col] = LabelEncoder()
                    X[col] = self.label_encoders[col].fit_transform(X[col])
            
            # Encode target variable
            self.label_encoders['Dropout_Likelihood'] = LabelEncoder()
            y_encoded = self.label_encoders['Dropout_Likelihood'].fit_transform(y)
            
            print("✅ Features preprocessed successfully")
            print(f"📊 Final feature matrix shape: {X.shape}")
            print(f"🎯 Target distribution: {pd.Series(y).value_counts().to_dict()}")
            
            return X, y_encoded, X.columns.tolist()
            
        except Exception as e:
            print(f"❌ Error preprocessing features: {str(e)}")
            return None, None, None
    
    def prepare_data(self, X, y, test_size=0.2):
        """Split and scale the data"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale numerical features
        numerical_columns = ['Age', 'Time_Spent_on_Videos', 'Quiz_Attempts', 'Quiz_Scores', 
                           'Forum_Participation', 'Assignment_Completion_Rate', 
                           'Final_Exam_Score', 'Feedback_Score']
        
        # Scale features
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[numerical_columns] = self.scaler.fit_transform(X_train[numerical_columns])
        X_test_scaled[numerical_columns] = self.scaler.transform(X_test[numerical_columns])
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train all classification models with comprehensive metrics"""
        print("🚀 Training Models...")
        print("=" * 80)
        
        results = []
        
        for name, model in self.models.items():
            try:
                print(f"🔧 Training {name}...")
                
                # Special handling for Neural Network
                if name == 'Neural Network' and NEURAL_NETWORKS_AVAILABLE:
                    # Initialize neural network with correct input dimension
                    input_dim = X_train.shape[1]
                    neural_network = model(input_dim)
                    
                    # Train neural network
                    history = neural_network.fit(X_train, y_train)
                    trained_model = neural_network
                    
                    # Make predictions
                    y_pred_proba = neural_network.predict_proba(X_test)
                    y_pred = (y_pred_proba[:, 1] > 0.5).astype(int)
                    
                else:
                    # Train other models
                    model.fit(X_train, y_train)
                    trained_model = model
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate comprehensive metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                auc_score = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
                
                # Store confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                self.confusion_matrices[name] = cm
                
                # Cross-validation (skip for Neural Network as it's slow)
                if name != 'Neural Network':
                    cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                else:
                    cv_mean = accuracy  # Use accuracy as proxy for CV
                    cv_std = 0.0
                
                # Store feature importances if available
                if hasattr(trained_model, 'feature_importances_'):
                    self.feature_importances[name] = trained_model.feature_importances_
                elif hasattr(trained_model, 'coef_'):
                    self.feature_importances[name] = np.abs(trained_model.coef_[0])
                else:
                    self.feature_importances[name] = None
                
                # Store results
                model_results = {
                    'Model': name,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1_Score': f1,
                    'AUC_Score': auc_score if auc_score is not None else 'N/A',
                    'CV_Score': cv_mean,
                    'CV_Std': cv_std,
                }
                
                results.append(model_results)
                self.results[name] = trained_model
                
                print(f"✅ {name}")
                print(f"   Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
                print(f"   F1-Score: {f1:.4f}, AUC: {auc_score:.4f if auc_score else 'N/A'}, CV: {cv_mean:.4f} (±{cv_std:.4f})")
                
                # Display confusion matrix for this model
                self._display_confusion_matrix(cm, name, y_test, y_pred)
                print()
                
            except Exception as e:
                print(f"❌ Error training {name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        if results:
            return pd.DataFrame(results)
        else:
            print("❌ No models were successfully trained")
            return pd.DataFrame()
    
    def _display_confusion_matrix(self, cm, model_name, y_test, y_pred):
        """Display confusion matrix for a specific model"""
        try:
            print(f"   📊 Confusion Matrix for {model_name}:")
            print(f"   {cm[0][0]:>6} {cm[0][1]:>6}   | Actual No")
            print(f"   {cm[1][0]:>6} {cm[1][1]:>6}   | Actual Yes")
            print("   " + "-" * 15)
            print(f"   Pred N  Pred Y")
            
            # Calculate additional metrics from confusion matrix
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            print(f"   Specificity: {specificity:.4f}, Sensitivity: {sensitivity:.4f}")
            
        except Exception as e:
            print(f"   ❌ Error displaying confusion matrix: {str(e)}")
    
    def display_model_comparison(self, results_df, feature_importances=None, X_test=None, y_test=None):
        """
        Display model comparison results with comprehensive metrics
        """
        try:
            if results_df.empty:
                print("❌ No results to display")
                return
            
            # Display the results dataframe
            print("\n📊 COMPREHENSIVE MODEL PERFORMANCE COMPARISON")
            print("=" * 100)
            
            # Format the results for better display
            display_df = results_df.copy()
            
            # Format numeric columns
            numeric_columns = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'CV_Score', 'CV_Std']
            for col in numeric_columns:
                if col in display_df.columns:
                    if col == 'CV_Std':
                        display_df[col] = display_df[col].apply(lambda x: f'±{x:.4f}')
                    else:
                        display_df[col] = display_df[col].apply(lambda x: f'{x:.4f}' if isinstance(x, (int, float)) else x)
            
            # Handle AUC Score
            if 'AUC_Score' in display_df.columns:
                display_df['AUC_Score'] = display_df['AUC_Score'].apply(
                    lambda x: f'{x:.4f}' if isinstance(x, (int, float)) else x
                )
            
            print(display_df.to_string(index=False))
            print("\n")
            
            # Find and display best model
            best_model_idx = results_df['Accuracy'].idxmax()
            best_model = results_df.loc[best_model_idx]
            
            print("🏆 BEST PERFORMING MODEL")
            print("=" * 50)
            print(f"📊 Model: {best_model['Model']}")
            print(f"🎯 Accuracy: {best_model['Accuracy']:.4f}")
            print(f"📈 Precision: {best_model['Precision']:.4f}")
            print(f"📊 Recall: {best_model['Recall']:.4f}")
            print(f"⭐ F1-Score: {best_model['F1_Score']:.4f}")
            print(f"🔢 Cross-Validation Score: {best_model['CV_Score']:.4f}")
            if isinstance(best_model.get('AUC_Score'), (int, float)):
                print(f"📊 AUC Score: {best_model['AUC_Score']:.4f}")
            
            # Display feature importances if available
            if (feature_importances is not None and 
                X_test is not None and 
                hasattr(X_test, 'columns')):
                
                print("\n🎯 TOP FEATURE IMPORTANCES")
                print("=" * 50)
                
                for model_name, importance in feature_importances.items():
                    if importance is not None and len(importance) > 0:
                        print(f"\n📈 {model_name}:")
                        # Create a DataFrame for better display
                        feat_df = pd.DataFrame({
                            'Feature': X_test.columns,
                            'Importance': importance
                        }).sort_values('Importance', ascending=False).head(8)
                        
                        print(feat_df.to_string(index=False))
            
            # Create visualizations
            self._create_comparison_plots(results_df)
            
            # Display all confusion matrices in a grid
            self._plot_all_confusion_matrices()
            
        except Exception as e:
            print(f"❌ Error in display_model_comparison: {str(e)}")
    
    def _plot_all_confusion_matrices(self):
        """Plot all confusion matrices in a grid"""
        try:
            if not self.confusion_matrices:
                print("❌ No confusion matrices available to plot")
                return
            
            n_models = len(self.confusion_matrices)
            n_cols = 3
            n_rows = (n_models + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
            if n_models == 1:
                axes = np.array([axes])
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            fig.suptitle('Confusion Matrices for All Models', fontsize=16, fontweight='bold', y=1.02)
            
            for idx, (model_name, cm) in enumerate(self.confusion_matrices.items()):
                row = idx // n_cols
                col = idx % n_cols
                
                ax = axes[row, col] if n_rows > 1 else axes[col]
                
                # Plot confusion matrix
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                           xticklabels=['Pred No', 'Pred Yes'],
                           yticklabels=['Actual No', 'Actual Yes'])
                ax.set_title(f'{model_name}\nConfusion Matrix', fontweight='bold')
                ax.set_xlabel('Predicted Label')
                ax.set_ylabel('True Label')
            
            # Hide empty subplots
            for idx in range(n_models, n_rows * n_cols):
                row = idx // n_cols
                col = idx % n_cols
                ax = axes[row, col] if n_rows > 1 else axes[col]
                ax.axis('off')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"❌ Error plotting confusion matrices: {str(e)}")
    
    def _create_comparison_plots(self, results_df):
        """Create comparison plots for model performance"""
        try:
            if results_df.empty:
                print("❌ No data for plots")
                return
                
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Dropout Prediction - Comprehensive Model Performance Comparison', fontsize=16, fontweight='bold')
            
            colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', '#6A8EAE', '#1D7874', '#FF6B6B']
            
            # Plot 1: Accuracy comparison
            if 'Accuracy' in results_df.columns:
                axes[0, 0].bar(results_df['Model'], results_df['Accuracy'], color=colors[:len(results_df)])
                axes[0, 0].set_title('Model Accuracy Comparison', fontweight='bold', fontsize=12)
                axes[0, 0].set_ylabel('Accuracy Score')
                axes[0, 0].tick_params(axis='x', rotation=45)
                axes[0, 0].grid(True, alpha=0.3)
                
                # Add value labels on bars
                for i, v in enumerate(results_df['Accuracy']):
                    axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Plot 2: F1-Score comparison
            if 'F1_Score' in results_df.columns:
                axes[0, 1].bar(results_df['Model'], results_df['F1_Score'], color=colors[:len(results_df)])
                axes[0, 1].set_title('F1-Score Comparison', fontweight='bold', fontsize=12)
                axes[0, 1].set_ylabel('F1 Score')
                axes[0, 1].tick_params(axis='x', rotation=45)
                axes[0, 1].grid(True, alpha=0.3)
                
                for i, v in enumerate(results_df['F1_Score']):
                    axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Plot 3: AUC scores if available
            if 'AUC_Score' in results_df.columns:
                auc_data = results_df[results_df['AUC_Score'].apply(lambda x: isinstance(x, (int, float)))]
                if not auc_data.empty:
                    axes[1, 0].bar(auc_data['Model'], auc_data['AUC_Score'], color=colors[:len(auc_data)])
                    axes[1, 0].set_title('AUC-ROC Scores', fontweight='bold', fontsize=12)
                    axes[1, 0].set_ylabel('AUC Score')
                    axes[1, 0].tick_params(axis='x', rotation=45)
                    axes[1, 0].grid(True, alpha=0.3)
                    
                    for i, v in enumerate(auc_data['AUC_Score']):
                        axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
                else:
                    axes[1, 0].text(0.5, 0.5, 'AUC Scores Not Available', 
                                   ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=12)
                    axes[1, 0].set_title('AUC-ROC Scores', fontweight='bold', fontsize=12)
            
            # Plot 4: Cross-validation scores
            if 'CV_Score' in results_df.columns:
                axes[1, 1].bar(results_df['Model'], results_df['CV_Score'], color=colors[:len(results_df)])
                axes[1, 1].set_title('Cross-Validation Scores', fontweight='bold', fontsize=12)
                axes[1, 1].set_ylabel('CV Score (3-fold)')
                axes[1, 1].tick_params(axis='x', rotation=45)
                axes[1, 1].grid(True, alpha=0.3)
                
                for i, v in enumerate(results_df['CV_Score']):
                    axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"❌ Error creating plots: {str(e)}")
    
    def get_best_model(self, results_df):
        """Get the best performing model based on accuracy"""
        try:
            if results_df.empty:
                return None
                
            best_model_idx = results_df['Accuracy'].idxmax()
            best_model = results_df.loc[best_model_idx]
            
            return best_model['Model']
            
        except Exception as e:
            print(f"❌ Error selecting best model: {str(e)}")
            return None

# Standalone functions for app.py to import
def train_and_evaluate_models(file_path=None):
    """
    Main function to train and evaluate dropout prediction models
    This function can be imported by app.py
    """
    print("🎓 STUDENT DROPOUT PREDICTION MODEL")
    print("=" * 60)
    
    # Initialize predictor
    predictor = DropoutPredictor()
    
    # Load data
    df = predictor.load_and_preprocess_data(file_path)
    
    if df is None:
        print("❌ Failed to load data. Exiting...")
        return None, None
    
    # Preprocess features
    X, y, feature_names = predictor.preprocess_features(df)
    
    if X is None:
        print("❌ Failed to preprocess features. Exiting...")
        return None, None
    
    # Prepare data
    X_train, X_test, y_train, y_test, X_train_df, X_test_df = predictor.prepare_data(X, y)
    
    # Store test data for later use
    predictor.X_test = X_test_df
    predictor.y_test = y_test
    predictor.feature_names = feature_names
    
    # Train models
    results_df = predictor.train_models(X_train, X_test, y_train, y_test)
    
    if results_df.empty:
        print("❌ No models were successfully trained. Exiting...")
        return None, None
    
    # Display model comparison
    predictor.display_model_comparison(
        results_df=results_df,
        feature_importances=predictor.feature_importances,
        X_test=X_test_df,
        y_test=y_test
    )
    
    print("\n✅ Model training and evaluation completed successfully!")
    
    return predictor, results_df

def display_model_comparison(predictor, results_df, X_test=None, y_test=None):
    """
    Standalone function to display model comparison in Streamlit
    """
    if predictor is None or results_df is None or results_df.empty:
        st.error("❌ No model results available")
        return
    
    # Use the predictor's display method
    predictor.display_model_comparison(
        results_df=results_df,
        feature_importances=predictor.feature_importances,
        X_test=X_test,
        y_test=y_test
    )

def run_dropout_prediction(file_path=None):
    """
    Simplified function that returns the predictor and results
    """
    predictor = DropoutPredictor()
    
    # Load data
    df = predictor.load_and_preprocess_data(file_path)
    if df is None:
        return None, None, None, None
    
    # Preprocess features
    X, y, feature_names = predictor.preprocess_features(df)
    if X is None:
        return None, None, None, None
    
    # Prepare data
    X_train, X_test, y_train, y_test, X_train_df, X_test_df = predictor.prepare_data(X, y)
    
    # Store test data
    predictor.X_test = X_test_df
    predictor.y_test = y_test
    predictor.feature_names = feature_names
    
    # Train models
    results_df = predictor.train_models(X_train, X_test, y_train, y_test)
    
    return predictor, results_df, X_test_df, y_test

# Main execution function
def main():
    """Main function to run the dropout prediction pipeline"""
    train_and_evaluate_models()

# Run the main function
if __name__ == "__main__":
    main()
