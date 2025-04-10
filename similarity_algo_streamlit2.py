import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import altair as alt

# Page title
st.title("Job-FDM Curriculum Similarity Matcher")

# Upload CSV
uploaded_file = st.file_uploader("Upload your job data CSV", type="csv")

# If a file is uploaded
if uploaded_file:
    job_data = pd.read_csv(uploaded_file)
    st.success("CSV uploaded successfully!")

    # Preview data
    st.subheader("Preview of Uploaded Job Data")
    st.write(job_data.head())

    # Check if 'Job Description' column exists
    if 'Job Description' in job_data.columns:
        # Define FDM Curriculum string
        fdm_curriculum = (
            "Software Development, Software Testing, DevOps, Cloud Computing, "
            "Site Reliability Engineering, Business Intelligence, Business Analytics, "
            "Project Support, Data Engineering, Data Science, Machine Learning, "
            "Data Governance, Technical Analysis, Amazon Web Services, Cyber Security, "
            "Risk and Compliance" 
        )

        # Remove missing descriptions
        job_req = job_data["Job Description"].dropna().tolist()

        # Combine job descriptions and curriculum
        all_texts = job_req + [fdm_curriculum]

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_texts)

        # Calculate cosine similarity
        similarity_scores = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])[0]
        ranked_indices = np.argsort(similarity_scores)[::-1]

        # Build results DataFrame
        ranked_jobs = pd.DataFrame({
            "Job Title": job_data["Title"].iloc[ranked_indices].values,
            "Company Name": job_data["Company"].iloc[ranked_indices].values,
            "Similarity Score": similarity_scores[ranked_indices]
        })

        # Show top results
        st.subheader("Top Matching Jobs to FDM Curriculum")
        st.dataframe(ranked_jobs.head(10))

        # Optional: Download results
        csv = ranked_jobs.to_csv(index=False)
        st.download_button("Download Full Ranked Jobs CSV", csv, "ranked_jobs.csv", "text/csv")

        # Plot Top 10 Job Matches Bar Chart
        top10 = ranked_jobs.head(10)

        st.subheader("Top 10 Matching Jobs (Bar Chart)")

        bar_chart = alt.Chart(top10).mark_bar().encode(
            x=alt.X("Similarity Score:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("Job Title:N", sort='-x'),
            tooltip=["Company Name", "Similarity Score"]
        ).properties(width=600)
        st.altair_chart(bar_chart)

    else:
        st.error("CSV must contain a 'Job Description' column.")
else:
    st.info("Please upload a CSV file with job listings.")
