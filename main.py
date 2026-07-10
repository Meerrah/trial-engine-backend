from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()

# Enable CORS so your React frontend can securely talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Vercel URL here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── 1. SIMULATE AND TRAIN A ML MODEL ON ENGINE START ───
print("🤖 Initializing and training Random Forest model...")

# Generate a synthetic dataset of 2,000 patients to train our model
np.random.seed(42)
n_samples = 2000

# Features
ast = np.random.uniform(10, 80, n_samples)       # Liver enzyme (U/L)
crp = np.random.uniform(0.5, 10.0, n_samples)    # Inflammation (mg/L)
dosage = np.random.uniform(10, 150, n_samples)   # Drug dosage (mg)

# Establish a mathematical hidden rule for toxicity (Ground Truth)
# Toxicity triggers if combinations are high or dosage overwhelms the system
toxicity_score = (0.3 * (ast / 40)) + (0.4 * (crp / 3.0)) + (0.5 * (dosage / 50))
y = (toxicity_score > 2.2).astype(int)  # 1 = Toxic failure, 0 = Safe profile

X = pd.DataFrame({"ast": ast, "crp": crp, "dosage": dosage})

# Train the Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
print("✅ Machine Learning model successfully trained!")


# ─── 2. DEFINE THE API DATA INPUT FORMAT ───
class CohortInput(BaseModel):
    cohort_size: int
    baseline_ast: float
    baseline_crp: float
    target_dosage: float


# ─── 3. THE PREDICTION ENDPOINT ───
@app.post("/predict")
def predict_toxicity(data: CohortInput):
    # Simulate a cohort around the baseline parameters selected by the user's sliders
    np.random.seed(10)
    
    # Generate variations matching the cohort size
    sim_ast = np.random.normal(data.baseline_ast, 5, data.cohort_size)
    sim_crp = np.random.normal(data.baseline_crp, 0.5, data.cohort_size)
    sim_dosage = np.ones(data.cohort_size) * data.target_dosage
    
    # Create evaluation dataframe
    sim_X = pd.DataFrame({"ast": sim_ast, "crp": sim_crp, "dosage": sim_dosage})
    
    # Run the real Random Forest model predictions
    predictions = model.predict(sim_X)
    
    # Calculate performance metrics
    total_failed = int(np.sum(predictions))
    total_safe = data.cohort_size - total_failed
    pass_rate = round((total_safe / data.cohort_size) * 100, 1)
    
    # Extract structural feature importance from the model
    importances = model.feature_importances_
    
    return {
        "synthetic_pass_rate": pass_rate,
        "safe_profiles": total_safe,
        "failed_profiles": total_failed,
        "feature_importance": {
            "dosage": round(importances[2] * 100, 1),
            "ast": round(importances[0] * 100, 1),
            "crp": round(importances[1] * 100, 1)
        }
    }
