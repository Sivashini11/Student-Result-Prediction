import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

st.title("🎓 Student Result Prediction System")

# Load dataset
data = pd.read_csv("student_data.csv")

# Convert Result to numeric
data['Result'] = data['Result'].map({'Fail': 0, 'Pass': 1})

# Handle missing values
data['Previous_Score'] = data['Previous_Score'].fillna(data['Previous_Score'].mean())

# Features
X = data[['Hours_Studied', 'Attendance', 'Previous_Score']]
y = data['Result']

# Train model
model = LogisticRegression()
model.fit(X, y)

# ---- UI INPUT FORM ----
st.subheader("Enter Student Details")

hours = st.slider("Hours Studied", 0, 10, 4)
attendance = st.slider("Attendance (%)", 0, 100, 75)
prev_score = st.number_input("Previous Score (leave 0 if first exam)", 0, 100, 0)

# Prediction button
if st.button("Predict Result"):
    input_data = [[hours, attendance, prev_score]]
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Student will PASS")
    else:
        st.error("❌ Student will FAIL")

# ---- GRAPH VISUALIZATION ----
st.subheader("📊 Data Visualization")

# Convert result back for visualization
data['Result_Label'] = data['Result'].map({0: 'Fail', 1: 'Pass'})

fig, ax = plt.subplots()
ax.scatter(data['Hours_Studied'], data['Previous_Score'])
ax.set_xlabel("Hours Studied")
ax.set_ylabel("Previous Score")
ax.set_title("Study Hours vs Previous Score")

st.pyplot(fig)