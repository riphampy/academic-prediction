import streamlit as st
import pickle
import pandas as pd
import numpy as np
import re
from scipy.sparse import hstack

# =============================
# Load model
# =============================
with open("model.pkl", "rb") as f:
    bundle = pickle.load(f)

models = bundle["models"]
tfidf = bundle["tfidf"]
alpha = bundle["alpha"]
num_features = bundle["num_features"]

# =============================
# Text cleaning 
# =============================
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# =============================
# UI
# =============================
st.title("🎓 Đoán mức độ cảnh báo học vụ")

age = st.number_input("Tuổi", 15, 60, 20)
tuition_debt = st.number_input("Nợ học phí", 0.0, 100000000.0, 0.0)
count_f = st.number_input("Số môn trượt", 0, 20, 0)
training_score = st.number_input("Điểm bài kiếm tra", 0.0, 100.0, 50.0)

if st.button("Predict"):

    # ===== Create DataFrame =====
    input_df = pd.DataFrame({
        "Age": [age],
        "Tuition_Debt": [tuition_debt],
        "Count_F": [count_f],
        "Training_Score_Mixed": [training_score],
    })

    # Add engineered features 
    input_df["num_subjects_taken"] = 5
    input_df["attendance_mean"] = 10
    input_df["low_attendance_count"] = 1
    input_df["zero_attendance_count"] = 0
    input_df["attendance_ratio"] = 10/16
    input_df["attendance_score"] = 10*5


    # ===== Ensemble predict =====
    probs = np.mean(
        [m.predict_proba(X) for m in models],
        axis=0
    )

    # Apply alpha
    probs[:, 2] *= alpha
    probs = probs / probs.sum(axis=1, keepdims=True)

    pred = np.argmax(probs, axis=1)[0]

    st.success(f"Predicted Academic Status: {pred}")
