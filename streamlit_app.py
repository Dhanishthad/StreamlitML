import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

st.title("📄 Resume Screening Based on Job Description")

# Sample Data
st.info("This app screens resumes based on job descriptions")

# Load resumes dataset (This is an example. You'll need to preprocess your own resume data)
df = pd.DataFrame({
    'Resume': [
        "I am a software engineer with 5 years of experience in Python and Machine Learning.",
        "Experienced data analyst with a background in SQL, Excel, and Power BI.",
        "I have strong experience in web development using JavaScript, React, and Node.js.",
        "Marketing professional skilled in digital campaigns, SEO, and content creation."
    ],
    'Job Fit': [1, 0, 1, 0]  # 1: Good fit, 0: Not a good fit
})

st.write("**Sample Resumes**")
st.dataframe(df)

# Input job description from the user
job_description = st.text_area("Enter the Job Description")

# Feature Engineering: Vectorize resumes and job descriptions using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
X = vectorizer.fit_transform(df['Resume']).toarray()
y = df['Job Fit']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

# Classification report
st.write("**Classification Report**")
st.text(classification_report(y_test, y_pred))

# Vectorize the input job description and make a prediction
if job_description:
    job_desc_vectorized = vectorizer.transform([job_description]).toarray()
    prediction = clf.predict(job_desc_vectorized)
    prediction_proba = clf.predict_proba(job_desc_vectorized)

    st.subheader("Prediction:")
    if prediction == 1:
        st.success("This resume is a good fit for the job!")
    else:
        st.error("This resume may not be a good fit for the job.")

    st.subheader("Prediction Probabilities:")
    st.write(f"Probability of Good Fit: {prediction_proba[0][1]:.2f}")
    st.write(f"Probability of Not a Good Fit: {prediction_proba[0][0]:.2f}")
