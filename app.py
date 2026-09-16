import streamlit as st
import pandas as pd
import joblib

# 1. Page Config (Thoda aur premium look ke liye icon add kiya)
st.set_page_config(page_title="HAR Project", page_icon="🏃‍♂️", layout="centered")

# 2. Sidebar hamesha upar rakhein
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2936/2936886.png", width=100) 
    st.title("Project Overview")
    st.info("""
    **Human Activity Recognition (HAR)**
    
    Yeh machine learning model smartphone ke accelerometer aur gyroscope data ko analyze karke user ki physical activity predict karta hai.
    
    * **Algorithm:** Random Forest
    * **Features:** 561 Sensor Readings
    * **Accuracy:** 91.14%
    """)
    st.markdown("---")
    st.write("👨‍💻 Developed by **Hiral**")

# 3. Main Page Title
st.title("🏃‍♂️ Human Activity Recognition")
st.markdown("Upload smartphone sensor data to instantly predict the user's physical activity.")

# 4. Model Load Karna
@st.cache_resource
def load_model():
    return joblib.load('har_model.pkl')

model = load_model()

# Model ke numbers ko asli naam me convert karne ke liye dictionary
activity_dict = {
    1: "WALKING 🚶",
    2: "WALKING UPSTAIRS 🧗",
    3: "WALKING DOWNSTAIRS 🚶‍♂️",
    4: "SITTING 🪑",
    5: "STANDING 🧍",
    6: "LAYING 🛌"
}

# 5. File Upload aur Processing
st.markdown("### 📂 Data Input")
uploaded_file = st.file_uploader("Upload Sensor Data (CSV format)", type="csv")

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file, header=None)
    
    # NAYA UI: Expander (Data chhupane aur click par dikhane ke liye)
    with st.expander("🔍 View Uploaded Raw Data (Click to expand)"):
        st.dataframe(input_data.head(5), use_container_width=True)
    
    # 6. Raw Data aur Waveform ka Graph Dikhana
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔢 Raw Values (First 10)")
        st.dataframe(input_data.iloc[0, :10].T, use_container_width=True) 
        
    with col2:
        st.markdown("#### 📈 Sensor Waveform")
        st.line_chart(input_data.iloc[0])
        
    st.markdown("---") # Ek divider line
    
    # 7. Prediction aur Probability
    # NAYA UI: Bada aur wide button
    if st.button("🚀 Predict Activity", use_container_width=True):
        
        # NAYA UI: Loading Spinner (Jab model soch raha ho)
        with st.spinner('Analyzing sensor patterns...'):
            prediction = model.predict(input_data.iloc[[0]])
            probabilities = model.predict_proba(input_data.iloc[[0]])[0]
            
            pred_num = prediction[0]
            activity_name = activity_dict.get(pred_num, "Unknown Activity")
            
            st.markdown("### 🎯 Prediction Result")
            
            # NAYA UI: Smart Metric Card
            st.metric(label="Detected Physical Activity", value=activity_name, delta="High Confidence")
            
            # NAYA UI: Dynamic Status Messages based on Activity
            if pred_num in [1, 2, 3]:
                st.success(f"**Status:** User is in motion ({activity_name})")
            elif pred_num in [4, 5]:
                st.info(f"**Status:** User is stationary ({activity_name})")
            elif pred_num == 6:
                st.warning(f"**Status:** User is resting ({activity_name})")
            
            st.markdown("#### 📊 Confidence Probability Chart")
            
            # Model actual me kin classes ki probab de raha hai
            model_classes = model.classes_
            class_names = [activity_dict.get(cls, f"Activity {cls}") for cls in model_classes]
            
            prob_df = pd.DataFrame({
                "Activity": class_names,
                "Probability (%)": probabilities * 100
            })
            
            st.bar_chart(prob_df, x="Activity", y="Probability (%)", color="#00a8e8")