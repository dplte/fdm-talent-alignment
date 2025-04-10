import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import altair as alt
from collections import Counter
import re

# Helper function to extract and count skill frequencies from descriptions
def extract_skill_frequencies(text):
    known_skills = [
        "python", "java", "c++", "aws", "azure", "sql", "tableau",
        "data visualization", "machine learning", "devops", "ci/cd",
        "gcp", "cloud computing", "business intelligence", "data science",
        "software development", "software testing"
    ]

    # Sort known skills by length (to match longer phrases first)
    known_skills.sort(key=lambda x: -len(x))

    # Count occurrences of skills in the text
    skill_freq = Counter()
    for skill in known_skills:
        pattern = re.escape(skill)
        matches = re.findall(pattern, text)
        if matches:
            skill_freq[skill] += len(matches)

    return skill_freq

# Function to extract key skills from a job description
def extract_skills(job_description):
    # List of common skills to detect
    known_skills = [
        "python", "java", "c++", "aws", "azure", "sql", "tableau",
        "data visualization", "machine learning", "devops", "ci/cd",
        "gcp", "cloud computing", "business intelligence", "data science",
        "software development", "software testing", "data engineering", "data governance"
    ]

    # Sort known skills by length (to match longer phrases first)
    known_skills.sort(key=lambda x: -len(x))

    # Find and return matching skills from the job description
    found_skills = []
    for skill in known_skills:
        if skill.lower() in job_description.lower():
            found_skills.append(skill)
    
    return ", ".join(found_skills) if found_skills else "No key skills detected"

# Streamlit page configuration
st.set_page_config(page_title="FDM Talent Alignment Platform", layout="wide")
st.title("FDM Talent Alignment Platform")

uploaded_file = st.file_uploader("Upload your job data CSV", type="csv")

if uploaded_file:
    job_data = pd.read_csv(uploaded_file)
    st.success("CSV uploaded successfully!")

    # Ensure expected columns exist
    if 'Job Description' not in job_data.columns or 'Title' not in job_data.columns or 'Company' not in job_data.columns:
        st.error("CSV must contain 'Job Description', 'Title', and 'Company' columns.")
    else:
        # FDM Curriculum
        fdm_curriculum = (
            "Software Development, Software Testing, DevOps, Cloud Computing, "
            "Site Reliability Engineering, Business Intelligence, Business Analytics, "
            "Project Support, Data Engineering, Data Science, Machine Learning, "
            "Data Governance, Technical Analysis, Amazon Web Services, Cyber Security, "
            "Risk and Compliance"
        )

        # Drop NaN values in Job Description
        job_data = job_data.dropna(subset=['Job Description'])

        # TF-IDF and Similarity Calculation
        job_req = job_data["Job Description"].tolist()
        all_texts = job_req + [fdm_curriculum]

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_texts)

        similarity_scores = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])[0]
        ranked_indices = np.argsort(similarity_scores)[::-1]

        # Add similarity scores
        job_data = job_data.iloc[ranked_indices].copy()
        job_data['Similarity Score'] = similarity_scores[ranked_indices]

        # Filters
        companies = job_data['Company'].unique()
        skills_filter = st.text_input("Filter by skill (e.g. Python, AWS, etc.)")
        company_filter = st.multiselect("Filter by Company", options=companies)

        filtered_data = job_data.copy()
        if company_filter:
            filtered_data = filtered_data[filtered_data['Company'].isin(company_filter)]

        if skills_filter:
            filtered_data = filtered_data[filtered_data['Job Description'].str.contains(skills_filter, case=False, na=False)]

        # Extract Key Skills
        filtered_data['Key Skills'] = filtered_data['Job Description'].apply(lambda x: extract_skills(x))

        # Top Matches Table
        st.subheader("Top Alignment Opportunities")
        st.dataframe(filtered_data[['Company', 'Title', 'Similarity Score', 'Job Description', 'Key Skills']].head(10))

        # Top Skill Demand
        st.subheader("Top Skill Demand")
        all_descriptions = " ".join(filtered_data['Job Description'].dropna()).lower()

        # Extract and count skill phrases
        skill_freq = extract_skill_frequencies(all_descriptions)
        skill_df = pd.DataFrame(skill_freq.items(), columns=["Skill", "Frequency"]).sort_values(by="Frequency", ascending=False)

        chart = alt.Chart(skill_df).mark_bar().encode(
            x=alt.X('Skill', sort='-y'),
            y='Frequency'
        ).properties(width=700, height=400)

        st.altair_chart(chart, use_container_width=True)

else:
    st.info("Please upload a CSV file with job listings.")











        



