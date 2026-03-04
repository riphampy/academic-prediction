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
tuition_debt = st.number_input(
    "Tiền nợ học phí (VND)",
    min_value=0,
    max_value=100000000,
    value=0,
    step=100000
)
count_f = st.number_input("Số môn trượt", 0, 20, 0)
training_score = st.number_input(
    "Điểm bài kiểm tra",
    min_value=0,
    max_value=100,
    value=50,
    step=1
)

advisor_notes = st.text_area("Nhận xét của cố vấn học tập")
personal_essay = st.text_area("Bài luận cá nhân")

if st.button("Predict"):

    # ===== Create DataFrame =====
    input_df = pd.DataFrame({
        "Age": [age],
        "Tuition_Debt": [tuition_debt],
        "Count_F": [count_f],
        "Training_Score_Mixed": [training_score],
    })

    # Add engineered features (đơn giản demo)
    input_df["num_subjects_taken"] = 5
    input_df["attendance_mean"] = 10
    input_df["low_attendance_count"] = 1
    input_df["zero_attendance_count"] = 0
    input_df["attendance_ratio"] = 10/16
    input_df["attendance_score"] = 10*5

    # ===== Text =====
    text_all = clean_text(advisor_notes) + " " + clean_text(personal_essay)
    input_df["text_all"] = text_all

    X_text = tfidf.transform(input_df["text_all"])
    X_tab = input_df[num_features]

    X = hstack([X_text, X_tab])

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
