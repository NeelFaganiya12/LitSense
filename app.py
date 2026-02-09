import streamlit as st
import json
import os
import random
from dotenv import load_dotenv
import google.generativeai as genai

# -----------------------------
# Load environment variables
# -----------------------------
from pathlib import Path
load_dotenv(dotenv_path=Path(".env"))

GEMINI_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3"),
]

GEMINI_API_KEY = random.choice([k for k in GEMINI_KEYS if k])
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("models/gemini-2.5-flash")

# import google.generativeai as genai

# genai.configure(api_key=GEMINI_API_KEY)

# # ---- DEBUG: list available models ----
# models = genai.list_models()
# st.write("Available models for this API key:")
# for m in models:
#     st.write(m.name)

# # Stop here so nothing else runs
# st.stop()

# -----------------------------
# Load papers from JSON
# -----------------------------
@st.cache_data
def load_papers():
    with open("papers.json", "r") as f:
        data = json.load(f)
    return data["references"]

papers = load_papers()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Literature Review Helper", layout="wide")

st.title("📚 Literature Review Assistant")
st.caption("Search, inspect, and shortlist papers for your literature review")

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("🔍 Search Filters")

search_query = st.sidebar.text_input(
    "Search (title, abstract, keywords)",
    placeholder="e.g. cognitive drift, memory systems"
)

min_year = min(p["year"] for p in papers)
max_year = max(p["year"] for p in papers)

year_range = st.sidebar.slider(
    "Publication year",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)

# -----------------------------
# Session state
# -----------------------------
if "selected_papers" not in st.session_state:
    st.session_state.selected_papers = []

if "ai_explanations" not in st.session_state:
    st.session_state.ai_explanations = {}

# -----------------------------
# Filtering logic
# -----------------------------
def paper_matches(paper):
    text = " ".join([
        paper["title"],
        paper["abstract"],
        " ".join(paper["keywords"])
    ]).lower()

    return (
        search_query.lower() in text
        and year_range[0] <= paper["year"] <= year_range[1]
    )

filtered_papers = [p for p in papers if paper_matches(p)]

# -----------------------------
# AI helper
# -----------------------------
def explain_relevance(paper, user_query=""):
    # Construct the search query with paper information
    authors_str = ", ".join(paper['authors'])
    paper_info_query = f"Explain the relevance of this paper: {paper['title']} by {authors_str} ({paper['year']})"
    
    # Combine with user's search query if provided
    if user_query and user_query.strip():
        full_query = f"{paper_info_query}. User's research interest: {user_query}"
    else:
        full_query = paper_info_query
    
    prompt = f"""
You are helping a graduate student with a literature review.

Search query:
"{full_query}"

Paper title:
"{paper['title']}"

Authors:
{", ".join(paper['authors'])}

Year:
{paper['year']}

Abstract:
"{paper['abstract']}"

Keywords:
{", ".join(paper['keywords'])}

In 3–4 sentences, explain the conceptual relevance of this paper based on the search query.
Focus on why this paper is relevant and what it contributes to the research area.
"""
    response = model.generate_content(prompt)
    return response.text.strip()

# -----------------------------
# Display papers
# -----------------------------
st.subheader(f"📄 Papers found: {len(filtered_papers)}")

for paper in filtered_papers:
    with st.expander(f"{paper['title']} ({paper['year']})"):
        st.markdown(f"**Authors:** {', '.join(paper['authors'])}")
        st.markdown(
            f"**Journal:** {paper['journal']}, "
            f"Vol {paper['volume']}({paper['issue']}), "
            f"pp. {paper['pages']}"
        )
        st.markdown(f"**DOI:** {paper['doi']}")
        st.markdown(f"**Keywords:** {', '.join(paper['keywords'])}")
        st.markdown("**Abstract:**")
        st.write(paper["abstract"])

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("➕ Add to reading list", key=f"add_{paper['id']}"):
                if paper not in st.session_state.selected_papers:
                    st.session_state.selected_papers.append(paper)

        with col2:
            if st.button("🤖 Explain relevance", key=f"ai_{paper['id']}"):
                with st.spinner("Asking Gemini..."):
                    try:
                        explanation = explain_relevance(paper, search_query)
                        st.session_state.ai_explanations[paper["id"]] = explanation
                    except Exception as e:
                        st.error(f"Gemini error: {str(e)}")

        with col3:
            if paper["id"] in st.session_state.ai_explanations:
                st.success("AI explanation ready")

        if paper["id"] in st.session_state.ai_explanations:
            st.markdown("### 🤖 Why this paper is relevant")
            st.write(st.session_state.ai_explanations[paper["id"]])

# -----------------------------
# Reading list section
# -----------------------------
st.divider()
st.subheader("📌 Reading List")

if st.session_state.selected_papers:
    for p in st.session_state.selected_papers:
        st.markdown(f"- **{p['title']}** ({p['year']}) — {p['journal']}")
else:
    st.caption("No papers selected yet.")
