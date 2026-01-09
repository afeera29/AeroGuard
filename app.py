# --- LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    # 1. Load the Model
    model = xgb.XGBRegressor()
    model.load_model("aeroguard_brain.json")
    
    # --- FIX FOR "ESTIMATOR TYPE UNDEFINED" ERROR ---
    # This manually tells Streamlit that "Yes, this is a regressor model"
    # It fixes the version mismatch bug.
    model._estimator_type = "regressor" 
    
    # 2. Load the "Lite" Data (The small file you just uploaded)
    # MAKE SURE this filename matches exactly what you uploaded to GitHub
    data = pd.read_csv("aeroguard_demo_data.csv")
    
    return model, data
