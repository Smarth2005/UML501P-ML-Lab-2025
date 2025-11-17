import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# ======================================================
# 1️⃣ Define Models
# ======================================================

# Anomaly Detection Models (A1-A8)
anomaly_models = {
    'A1': IsolationForest(contamination=0.1, random_state=123),
    'A2': LocalOutlierFactor(n_neighbors=20, contamination=0.1),
    'A3': OneClassSVM(nu=0.1),
    'A4': LocalOutlierFactor(n_neighbors=5, contamination=0.1),
    'A5': PCA(n_components=0.95),
}

# Regression Models (M1-M6)
regression_models = {
    'M1': DecisionTreeRegressor(random_state=123),
    'M2': RandomForestRegressor(n_estimators=100, random_state=123, n_jobs=-1),
    'M3': AdaBoostRegressor(random_state=123),
    'M4': GradientBoostingRegressor(random_state=123),
    'M5': Ridge(alpha=1.0),
    'M6': LinearRegression()
}

# ======================================================
# 2️⃣ Load and PREPROCESS Data (FIXED)
# ======================================================

def preprocess_data(df, target_column):
    """Preprocess data: handle missing values, encode categorical variables"""
    df_clean = df.copy()
    
    # Check if target exists
    if target_column not in df_clean.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    
    # Drop rows where target is missing
    df_clean = df_clean.dropna(subset=[target_column])
    
    # Separate features and target
    X = df_clean.drop(columns=[target_column])
    y = df_clean[target_column]
    
    print("🔍 Data Types in Features:")
    print(X.dtypes)
    print(f"\n🔍 Target ({target_column}) type: {y.dtype}")
    
    # Handle categorical columns in features
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    
    print(f"📊 Categorical columns: {list(categorical_cols)}")
    print(f"📊 Numeric columns: {list(numeric_cols)}")
    
    # Encode categorical variables
    label_encoders = {}
    X_encoded = X.copy()
    
    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        print(f"   Encoded {col}: {len(le.classes_)} categories")
    
    # Handle missing values in numeric columns
    imputer = SimpleImputer(strategy='median')
    if len(numeric_cols) > 0:
        X_encoded[numeric_cols] = imputer.fit_transform(X_encoded[numeric_cols])
    
    # Convert target to numeric if needed
    if not np.issubdtype(y.dtype, np.number):
        print(f"⚠️  Target column '{target_column}' is not numeric. Converting...")
        y = pd.to_numeric(y, errors='coerce')
        # Drop rows where target conversion failed
        valid_mask = ~y.isna()
        X_encoded = X_encoded[valid_mask]
        y = y[valid_mask]
        print(f"   Target converted to numeric. Remaining samples: {len(y)}")
    
    print(f"✅ Final dataset shape: {X_encoded.shape}")
    return X_encoded, y

# Load data
try:
    data = pd.read_excel("Anomaly Dataset.xlsx")
    print("✅ Dataset loaded from file")
    
    # Display basic info about the dataset
    print(f"📊 Original data shape: {data.shape}")
    print("\n📋 Column names:")
    print(data.columns.tolist())
    print(f"\n🔍 Data types:")
    print(data.dtypes)
    print(f"\n📝 First few rows:")
    print(data.head())
    
except Exception as e:
    print(f"⚠️  Using sample data for demo. File error: {e}")
    np.random.seed(123)
    n_samples = 1000
    data = pd.DataFrame({
        'Feature1': np.random.normal(0, 1, n_samples),
        'Feature2': np.random.normal(0, 1, n_samples),
        'Feature3': np.random.normal(0, 1, n_samples),
        'Feature4': np.random.normal(0, 1, n_samples),
        'CGPA': np.random.normal(3.0, 0.5, n_samples)
    })

target = 'CGPA'

# Preprocess data
try:
    X, y = preprocess_data(data, target)
    features = X.columns.tolist()
    
    print(f"\n🎯 Target: {target}")
    print(f"📋 Features ({len(features)}): {features}")
    print(f"📊 Final X shape: {X.shape}, y shape: {y.shape}")
    
except Exception as e:
    print(f"❌ Error in preprocessing: {e}")
    exit()

# ======================================================
# 3️⃣ Baseline (Without Anomaly Removal)
# ======================================================

print("\n🔧 Calculating Baseline (Without Anomaly Removal)...")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123
)

print(f"📊 Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

baseline_scores = {}

for m_name, model in regression_models.items():
    try:
        print(f"   Training {m_name}...", end=" ")
        # Train and predict
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Calculate R²
        r2 = r2_score(y_test, y_pred)
        baseline_scores[m_name] = round(r2, 4)
        print(f"R² = {r2:.4f}")
    except Exception as e:
        baseline_scores[m_name] = "N/A"
        print(f"Error - {e}")

# ======================================================
# 4️⃣ With Anomaly Removal (Optimized & Fixed)
# ======================================================

print("\n🔧 Calculating With Anomaly Removal...")

# Scale the entire dataset for anomaly detection
scaler_full = StandardScaler()
X_scaled = scaler_full.fit_transform(X)

results_with_anomaly = {}

for a_name, anomaly_model in anomaly_models.items():
    print(f"\n🔍 Processing {a_name}...")
    
    try:
        # Detect anomalies
        if a_name == 'A5':  # PCA-based anomaly detection
            pca_model = anomaly_model.fit(X_scaled)
            pca_scores = pca_model.transform(X_scaled)
            reconstructed = pca_model.inverse_transform(pca_scores)
            reconstruction_error = np.mean((X_scaled - reconstructed) ** 2, axis=1)
            # Mark top 10% as anomalies
            anomaly_threshold = np.percentile(reconstruction_error, 90)
            inlier_mask = reconstruction_error <= anomaly_threshold
            
        elif a_name == 'A3':  # OneClassSVM
            anomaly_model.fit(X_scaled)
            anomaly_labels = anomaly_model.predict(X_scaled)
            inlier_mask = anomaly_labels == 1
            
        else:  # IsolationForest, LOF
            anomaly_labels = anomaly_model.fit_predict(X_scaled)
            inlier_mask = anomaly_labels == 1
        
        # Get clean data
        X_clean = X.iloc[inlier_mask] if hasattr(inlier_mask, '__array_interface__') else X[inlier_mask]
        y_clean = y.iloc[inlier_mask] if hasattr(inlier_mask, '__array_interface__') else y[inlier_mask]
        
        print(f"   Clean data: {X_clean.shape[0]}/{X.shape[0]} samples")
        
        if len(X_clean) < 20:
            print("   ⚠️  Too few samples, skipping")
            results_with_anomaly[a_name] = {m: "N/A" for m in regression_models.keys()}
            continue
        
        # Split clean data
        X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(
            X_clean, y_clean, test_size=0.3, random_state=123
        )
        
        # Scale clean data
        scaler_clean = StandardScaler()
        X_train_clean_scaled = scaler_clean.fit_transform(X_train_clean)
        X_test_clean_scaled = scaler_clean.transform(X_test_clean)
        
        # Train regression models on clean data
        anomaly_scores = {}
        for m_name, model in regression_models.items():
            try:
                # Create new instance to avoid refitting issues
                model_clone = type(model)(**model.get_params())
                model_clone.fit(X_train_clean_scaled, y_train_clean)
                y_pred_clean = model_clone.predict(X_test_clean_scaled)
                
                r2_clean = r2_score(y_test_clean, y_pred_clean)
                anomaly_scores[m_name] = round(r2_clean, 4)
                print(f"   {m_name}: R² = {r2_clean:.4f}")
            except Exception as e:
                anomaly_scores[m_name] = "N/A"
                print(f"   {m_name}: Error - {e}")
        
        results_with_anomaly[a_name] = anomaly_scores
        
    except Exception as e:
        print(f"   ❌ Error in {a_name}: {e}")
        results_with_anomaly[a_name] = {m: "N/A" for m in regression_models.keys()}

# ======================================================
# 5️⃣ Create Final Comparison Table
# ======================================================

print("\n" + "="*80)
print("PREDICTION MODELS PERFORMANCE ON R-SQUARE")
print("="*80)

# Create results table
anomaly_techniques = list(anomaly_models.keys())
prediction_models = list(regression_models.keys())

# Initialize table
table_data = []

for a_name in anomaly_techniques:
    row = [a_name]
    
    # Without anomaly removal scores
    for m_name in prediction_models:
        row.append(baseline_scores.get(m_name, "N/A"))
    
    # With anomaly removal scores
    for m_name in prediction_models:
        row.append(results_with_anomaly.get(a_name, {}).get(m_name, "N/A"))
    
    table_data.append(row)

# Create DataFrame
columns = ['Anomaly Techniques']
columns.extend([f'Without_{m}' for m in prediction_models])
columns.extend([f'With_{m}' for m in prediction_models])

results_df = pd.DataFrame(table_data, columns=columns)
results_df.set_index('Anomaly Techniques', inplace=True)

# Display formatted table
print("\n" + " "*10 + "WITHOUT ANOMALY REMOVAL" + " "*15 + "WITH ANOMALY REMOVAL")
print(" " * 8 + "M1     M2     M3     M4     M5     M6" + "     " + "M1     M2     M3     M4     M5     M6")
print("-" * 80)

for a_name in anomaly_techniques:
    row_str = f"{a_name:>4}  "
    
    # Without scores
    for m_name in prediction_models:
        score = baseline_scores.get(m_name, "N/A")
        if score != "N/A":
            row_str += f"{score:>6.3f}" if isinstance(score, (int, float)) else f"{score:>6}"
        else:
            row_str += "   N/A"
    
    row_str += "   "
    
    # With scores
    for m_name in prediction_models:
        score = results_with_anomaly.get(a_name, {}).get(m_name, "N/A")
        if score != "N/A":
            row_str += f"{score:>6.3f}" if isinstance(score, (int, float)) else f"{score:>6}"
        else:
            row_str += "   N/A"
    
    print(row_str)

# ======================================================
# 6️⃣ Save Table as CSV
# ======================================================

print("\n💾 Saving table as CSV...")

# Create CSV with exact table format
csv_data = []

# Header row
header_row = ['Anomaly Techniques']
for m in prediction_models:
    header_row.append(f'Without_{m}')
for m in prediction_models:
    header_row.append(f'With_{m}')
csv_data.append(header_row)

# Data rows
for a_name in anomaly_techniques:
    row = [a_name]
    
    # Without anomaly removal scores
    for m_name in prediction_models:
        score = baseline_scores.get(m_name, "N/A")
        row.append(score)
    
    # With anomaly removal scores
    for m_name in prediction_models:
        score = results_with_anomaly.get(a_name, {}).get(m_name, "N/A")
        row.append(score)
    
    csv_data.append(row)

# Save to CSV
csv_df = pd.DataFrame(csv_data[1:], columns=csv_data[0])
csv_df.to_csv("R2_Comparison_Table.csv", index=False)
print("✅ Table saved as 'R2_Comparison_Table.csv'")

# ======================================================
# 7️⃣ PRINT TOP 3 MODELS (FAST)
# ======================================================

print("\n" + "="*40)
print("TOP 3 BEST PERFORMING COMBINATIONS")
print("="*40)

# Quick collection of all scores
all_scores = []

# Add baseline scores
for m_name, score in baseline_scores.items():
    if score != "N/A" and isinstance(score, (int, float)):
        all_scores.append((score, m_name, "Baseline"))

# Add anomaly removal scores
for a_name, model_scores in results_with_anomaly.items():
    for m_name, score in model_scores.items():
        if score != "N/A" and isinstance(score, (int, float)):
            all_scores.append((score, m_name, a_name))

# Get top 3
if all_scores:
    top_3 = sorted(all_scores, key=lambda x: x[0], reverse=True)[:3]
    
    print("\n📋 SUBMISSION FORMAT:")
    print("-" * 25)
    for score, model, anomaly in top_3:
        if anomaly == "Baseline":
            print(f"R_Square Value = {model}")
        else:
            print(f"R_Square Value = {model} + {anomaly}")
else:
    print("❌ No valid combinations found!")

print("\n✅ Analysis Complete!")