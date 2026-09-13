import streamlit as st
import pandas as pd
import joblib

# 1. Page Config sabse pehle aani chahiye
st.set_page_config(page_title="Human Activity Recognition", layout="centered")

# 2. Sidebar hamesha upar rakhein
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2936/2936886.png", width=100) # Ek dummy icon
    st.title("About Project")
    st.info("""
    **Human Activity Recognition**
    
    Yeh model smartphone ke accelerometer aur gyroscope ke data ko analyse karke predict karta hai ki user konsi activity kar raha hai.
    
    **Algorithm:** Random Forest
    **Features:** 561 Sensor Readings
    """)
    st.write("Developed by Hiral")

# 3. Main Page Title
st.title("🏃‍♂️ Human Activity Recognition (HAR)")

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
uploaded_file = st.file_uploader("Upload Sensor Data (CSV)", type="csv")

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file, header=None)
    st.dataframe(input_data.head(1))
    
    # 6. Raw Data aur Waveform ka Graph Dikhana
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Raw Sensor Data (First 10 values):")
        st.dataframe(input_data.iloc[0, :10].T, use_container_width=True) 
        
    with col2:
        st.write("### Sensor Waveform:")
        st.line_chart(input_data.iloc[0])
        
    # 7. Prediction aur Probability
    if st.button("Predict Activity"):
        # Prediction aur Probability nikalna
        prediction = model.predict(input_data.iloc[[0]])
        probabilities = model.predict_proba(input_data.iloc[[0]])[0]
        
        pred_num = prediction[0]
        activity_name = activity_dict.get(pred_num, "Unknown Activity")
        
        st.success(f"### Predicted Activity: **{activity_name}**")
        
        st.write("#### Model Confidence:")
        
        # Model actual me kin classes (1,2,3,4,5,6) ki probab de raha hai, usko access karna
        model_classes = model.classes_
        
        # Un classes ke naam nikalna dictionary se (taaki dono ki length hamesha match ho)
        class_names = [activity_dict.get(cls, f"Activity {cls}") for cls in model_classes]
        
        prob_df = pd.DataFrame({
            "Activity": class_names,
            "Probability": probabilities * 100
        })
        
        st.bar_chart(prob_df, x="Activity", y="Probability", color="#00a8e8")