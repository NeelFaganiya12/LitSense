import streamlit as st
import json
import os
import random
from dotenv import load_dotenv

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

GEMINI_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3"),
]
GEMINI_API_KEY = random.choice(GEMINI_KEYS)

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

search_query = st.sidebar.text_input("Search (title, abstract, keywords)")

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
# Display papers
# -----------------------------
st.subheader(f"📄 Papers found: {len(filtered_papers)}")

for paper in filtered_papers:
    with st.expander(f"{paper['title']} ({paper['year']})"):
        st.markdown(f"**Authors:** {', '.join(paper['authors'])}")
        st.markdown(f"**Journal:** {paper['journal']}, Vol {paper['volume']}({paper['issue']}), pp. {paper['pages']}")
        st.markdown(f"**DOI:** {paper['doi']}")
        st.markdown(f"**Keywords:** {', '.join(paper['keywords'])}")
        st.markdown("**Abstract:**")
        st.write(paper["abstract"])

        col1, col2 = st.columns(2)

        with col1:
            if st.button("➕ Add to reading list", key=f"add_{paper['id']}"):
                st.session_state.selected_papers.append(paper)

        with col2:
            if st.button("🤖 AI relevance (stub)", key=f"ai_{paper['id']}"):
                st.info(
                    "Gemini call would go here.\n\n"
                    f"Using API key: `{GEMINI_API_KEY[:5]}...`"
                )

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
