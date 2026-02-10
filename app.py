import streamlit as st
import json
import os
import random
import requests
import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from collections import defaultdict
import re

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

# -----------------------------
# Semantic Scholar API functions
# -----------------------------
def search_semantic_scholar(query: str, limit: int = 20, year_filter: str = None, min_citations: int = None):
    """
    Search Semantic Scholar API for papers using the official endpoint
    Documentation: https://api.semanticscholar.org/api-docs/graph#tag/Paper-Data/operation/get_graph_paper_relevance_search
    """
    query = query.strip()
    if not query:
        return []
    
    # Clean query - remove redundant words
    query = query.replace(" papers", "").replace(" paper", "")
    query = query.replace(" articles", "").replace(" article", "")
    query = query.strip()
    
    # Ensure limit is within valid range (max 100 per API docs)
    limit = max(1, min(limit, 100))
    
    # Use the correct endpoint from documentation
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    
    # Build parameters according to API documentation
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,authors,year,abstract,venue,citationCount,url,externalIds,fieldsOfStudy,publicationTypes"
    }
    
    # Add optional filters
    if year_filter:
        params["year"] = year_filter  # Format: "2016-2020" or "2019"
    
    if min_citations:
        params["minCitationCount"] = str(min_citations)
    
    try:
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # API returns: total, offset, next, data[]
            papers_data = data.get('data', [])
            total_results = data.get('total', 0)
            
            papers = []
            for idx, paper in enumerate(papers_data, 1):
                # Extract authors - API returns array with name field
                authors = []
                if paper.get('authors'):
                    authors = [author.get('name', '') for author in paper.get('authors', []) if author.get('name')]
                
                # Extract DOI from externalIds
                doi = ""
                if paper.get('externalIds') and isinstance(paper.get('externalIds'), dict):
                    doi = paper.get('externalIds', {}).get('DOI', '')
                
                # Extract fields of study
                fields_of_study = paper.get('fieldsOfStudy', [])
                
                paper_obj = {
                    "id": idx,
                    "paperId": paper.get('paperId', ''),
                    "title": paper.get('title', 'Untitled'),
                    "authors": authors if authors else ["Unknown"],
                    "year": paper.get('year') or 0,
                    "abstract": paper.get('abstract', 'No abstract available'),
                    "journal": paper.get('venue', 'N/A'),
                    "doi": doi,
                    "keywords": fields_of_study,  # Use fields of study as keywords
                    "citation_count": paper.get('citationCount', 0),
                    "url": paper.get('url', ''),
                    "fieldsOfStudy": fields_of_study,
                    "publicationTypes": paper.get('publicationTypes', [])
                }
                papers.append(paper_obj)
            
            return papers
        
        elif response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 300))
            st.session_state.rate_limit_time = datetime.datetime.now() + datetime.timedelta(seconds=retry_after)
            st.error(f"""
            ❌ **Rate Limit Exceeded (429)**
            
            Please wait {retry_after // 60} minutes before trying again.
            """)
            return []
        
        elif response.status_code == 400:
            st.error("❌ Invalid search query. Try a different search term.")
            return []
        
        else:
            st.error(f"❌ API error: {response.status_code}")
            return []
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return []

# -----------------------------
# AI Clustering function
# -----------------------------
def cluster_papers(papers, search_query):
    """Use AI to cluster papers into thematic groups"""
    if not papers or not GEMINI_API_KEY:
        return {}
    
    try:
        # Create summary of papers
        papers_summary = "\n\n".join([
            f"Paper {i+1}: {p['title']} - {p['abstract'][:150]}..."
            for i, p in enumerate(papers[:15])
        ])
        
        prompt = f"""Given these research papers about "{search_query}", organize them into 3-5 thematic clusters.

Papers:
{papers_summary}

For each cluster, provide:
1. A short cluster name (2-4 words)
2. The paper numbers that belong to it
3. Key topics/keywords for that cluster

Format:
CLUSTER 1: [name]
Papers: [numbers]
Topics: [keywords]

CLUSTER 2: [name]
Papers: [numbers]
Topics: [keywords]
..."""
        
        response = model.generate_content(prompt)
        text = response.text
        
        # Parse clusters
        clusters = {}
        current_cluster = None
        
        for line in text.split('\n'):
            if 'CLUSTER' in line.upper() or 'CLUSTER' in line:
                # Extract cluster name
                parts = line.split(':', 1)
                if len(parts) > 1:
                    current_cluster = parts[1].strip()
                    clusters[current_cluster] = {"papers": [], "topics": []}
            elif current_cluster and 'Papers:' in line:
                # Extract paper numbers
                numbers = re.findall(r'\d+', line)
                clusters[current_cluster]["papers"] = [int(n) - 1 for n in numbers if int(n) <= len(papers)]
            elif current_cluster and 'Topics:' in line:
                # Extract topics
                topics = line.split('Topics:')[1].strip()
                clusters[current_cluster]["topics"] = [t.strip() for t in topics.split(',')[:5]]
        
        return clusters
    except Exception as e:
        return {}

# -----------------------------
# AI helper - Explain relevance
# -----------------------------
def explain_relevance(paper, user_query=""):
    """Use Gemini to explain paper relevance"""
    authors_str = ", ".join(paper.get('authors', ['Unknown']))
    paper_info_query = f"Explain the relevance of this paper: {paper.get('title', 'N/A')} by {authors_str} ({paper.get('year', 'N/A')})"
    
    if user_query and user_query.strip():
        full_query = f"{paper_info_query}. User's research interest: {user_query}"
    else:
        full_query = paper_info_query
    
    prompt = f"""How is this paper relevant here?

Paper title: {paper.get('title', 'N/A')}
Authors: {', '.join(paper.get('authors', ['Unknown']))}
Year: {paper.get('year', 'N/A')}
Journal: {paper.get('journal', 'N/A')}

Abstract:
{paper.get('abstract', 'No abstract available')}

User's search topic: {user_query if user_query else 'General research'}

Please explain in 3-4 sentences how this paper is relevant to the user's search topic.
Focus on conceptual relevance and what this paper contributes to the research area.
"""
    response = model.generate_content(prompt)
    return response.text.strip()

# -----------------------------
# AI Relevance Ranking
# -----------------------------
def rank_papers_by_relevance(papers, search_query):
    """Rank papers by relevance to search query"""
    if not papers or not GEMINI_API_KEY:
        return papers
    
    try:
        papers_list = "\n".join([
            f"{i+1}. {p['title']}"
            for i, p in enumerate(papers[:10])
        ])
        
        prompt = f"""Rank these papers by relevance to "{search_query}" (most relevant first).

Papers:
{papers_list}

Return only the numbers in order of relevance, separated by commas."""
        
        response = model.generate_content(prompt)
        ranked_indices = [int(x.strip()) - 1 for x in response.text.split(',') if x.strip().isdigit()]
        
        # Reorder papers
        ranked = [papers[i] for i in ranked_indices if 0 <= i < len(papers)]
        # Add any papers not in ranking
        remaining = [p for i, p in enumerate(papers) if i not in ranked_indices]
        return ranked + remaining
    except:
        return papers

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Literature Review Assistant", layout="wide", initial_sidebar_state="collapsed")

# Initialize session state
if "selected_papers" not in st.session_state:
    st.session_state.selected_papers = []
if "ai_explanations" not in st.session_state:
    st.session_state.ai_explanations = {}
if "cached_papers" not in st.session_state:
    st.session_state.cached_papers = {}
if "last_search_query" not in st.session_state:
    st.session_state.last_search_query = ""
if "rate_limit_time" not in st.session_state:
    st.session_state.rate_limit_time = None
if "selected_paper_id" not in st.session_state:
    st.session_state.selected_paper_id = None
if "paper_feedback" not in st.session_state:
    st.session_state.paper_feedback = {}
if "clusters" not in st.session_state:
    st.session_state.clusters = {}
if "ranked_papers" not in st.session_state:
    st.session_state.ranked_papers = []
if "last_data_source" not in st.session_state:
    st.session_state.last_data_source = None
if "all_loaded_papers" not in st.session_state:
    st.session_state.all_loaded_papers = []  # Store all papers that have been loaded for feedback tracking
if "current_papers" not in st.session_state:
    st.session_state.current_papers = []  # Store current papers for paper details access

# Header
col_header1, col_header2 = st.columns([4, 1])
with col_header1:
    st.title("📚 Literature Review Assistant")
    st.caption("Search → Explore → Review → Refine")
with col_header2:
    if st.session_state.selected_papers:
        st.metric("Saved", len(st.session_state.selected_papers))

# Main layout: 3 columns
col_search, col_results, col_details = st.columns([1, 2, 1.5])

# ==================== LEFT COLUMN: SEARCH & STEERING ====================
with col_search:
    st.header("🔍 Search & Filter")
    
    # Data source selection
    data_source = st.radio(
        "Data Source",
        ["Semantic Scholar API", "Local Papers"],
        help="Choose between live search or local papers.json file",
        key="data_source_selector"
    )
    
    # Note about loading time
    st.caption("⏱️ Note: Processing may take 10-15 seconds (AI clustering & ranking)")
    
    # Clear selected paper when switching data sources
    if data_source != st.session_state.get('last_data_source'):
        if st.session_state.get('last_data_source') is not None:
            st.session_state.selected_paper_id = None
            # Clear AI explanations when switching to avoid showing wrong explanations
            st.session_state.ai_explanations = {}
    
    st.divider()
    
    if data_source == "Semantic Scholar API":
        search_query = st.text_input(
            "Research Topic",
            placeholder="e.g. machine learning, transformer architectures",
            help="Enter your research topic to search Semantic Scholar",
            key="main_search"
        )
        
        # Search button
        search_clicked = st.button("🔎 Search", type="primary", use_container_width=True)
    else:
        # Local papers mode
        col_filter1, col_filter2 = st.columns([3, 1])
        with col_filter1:
            search_query = st.text_input(
                "Filter Papers",
                placeholder="Search within local papers...",
                help="Filter papers from papers.json by title, abstract, or keywords",
                key="local_search"
            )
        with col_filter2:
            # Show clear button only if filter is active
            if st.session_state.get('local_search', ''):
                if st.button("🗑️ Clear", use_container_width=True, help="Clear the filter"):
                    st.session_state.local_search = ""
                    st.rerun()
            else:
                st.write("")  # Empty space to maintain layout
        
        search_clicked = st.button("🔍 Filter", type="primary", use_container_width=True)
    
    # Check rate limit (only for Semantic Scholar)
    if data_source == "Semantic Scholar API" and st.session_state.rate_limit_time:
        time_remaining = (st.session_state.rate_limit_time - datetime.datetime.now()).total_seconds()
        if time_remaining > 0:
            st.warning(f"⏰ Rate limited: {int(time_remaining // 60)}m {int(time_remaining % 60)}s")
        else:
            st.session_state.rate_limit_time = None
    
    st.divider()
    
    # Year filter (only for Semantic Scholar)
    if data_source == "Semantic Scholar API":
        year_filter_enabled = st.checkbox("Filter by Year", value=False)
        if year_filter_enabled:
            year_range = st.slider(
                "Publication Year",
                min_value=2000,
                max_value=2025,
                value=(2020, 2025)
            )
            st.session_state.year_filter = year_range
        else:
            if "year_filter" in st.session_state:
                del st.session_state.year_filter
    
    st.divider()
    
    # Cached searches (only for Semantic Scholar)
    if data_source == "Semantic Scholar API" and st.session_state.cached_papers:
        st.subheader("📦 Recent Searches")
        for cached_query in list(st.session_state.cached_papers.keys())[:5]:
            if st.button(f"📄 {cached_query[:30]}...", key=f"cache_{cached_query}", use_container_width=True):
                st.session_state.last_search_query = cached_query
                st.rerun()
    
    # Show local papers info
    if data_source == "Local Papers":
        st.info("💡 **Local Papers Mode**\n\nShowing papers from `papers.json` file. Use the filter box above to search within local papers.")

# ==================== MIDDLE COLUMN: RESULTS (CLUSTERS & QUEUE) ====================
with col_results:
    papers = []
    
    # Load papers based on data source
    if data_source == "Local Papers":
        # Load from local JSON file
        @st.cache_data
        def load_local_papers():
            try:
                with open("papers.json", "r") as f:
                    data = json.load(f)
                local_papers = data.get("references", [])
                
                # Normalize local papers to match expected format
                normalized = []
                for idx, p in enumerate(local_papers, 1):
                    normalized_paper = {
                        "id": p.get('id', idx),
                        "title": p.get('title', 'Untitled'),
                        "authors": p.get('authors', ['Unknown']),
                        "year": p.get('year', 0),
                        "abstract": p.get('abstract', 'No abstract available'),
                        "journal": p.get('journal', 'N/A'),
                        "doi": p.get('doi', ''),
                        "keywords": p.get('keywords', []),
                        "citation_count": 0,  # Local papers don't have citation data
                        "url": "",
                        "volume": p.get('volume', ''),
                        "issue": p.get('issue', ''),
                        "pages": p.get('pages', '')
                    }
                    normalized.append(normalized_paper)
                return normalized
            except FileNotFoundError:
                st.error("❌ papers.json file not found!")
                return []
            except json.JSONDecodeError as e:
                st.error(f"❌ Invalid JSON in papers.json file: {str(e)}")
                return []
        
        all_local_papers = load_local_papers()
        
        # Filter local papers if search query provided
        # Get the actual search query value from session state
        local_search_value = st.session_state.get('local_search', '')
        if local_search_value:
            query_lower = local_search_value.lower()
            papers = [
                p for p in all_local_papers
                if (query_lower in p.get('title', '').lower() or
                    query_lower in p.get('abstract', '').lower() or
                    any(query_lower in kw.lower() for kw in p.get('keywords', [])))
            ]
        else:
            papers = all_local_papers
        
        # Generate clusters and ranking for local papers (only if papers loaded)
        if papers:
            # Regenerate clusters/ranking when switching to local papers or on new search
            local_search_value = st.session_state.get('local_search', '')
            if data_source != st.session_state.get('last_data_source') or search_clicked or not st.session_state.clusters:
                with st.spinner("🤖 Organizing papers into clusters..."):
                    cluster_query = local_search_value if local_search_value else "research papers"
                    st.session_state.clusters = cluster_papers(papers, cluster_query)
                
                with st.spinner("📊 Ranking papers by relevance..."):
                    rank_query = local_search_value if local_search_value else "general research"
                    st.session_state.ranked_papers = rank_papers_by_relevance(papers, rank_query)
            
            st.session_state.last_data_source = data_source
    
    else:
        # Semantic Scholar API mode
        # Get year filter if set
        year_filter = None
        if "year_filter" in st.session_state:
            year_range = st.session_state.year_filter
            if isinstance(year_range, (list, tuple)) and len(year_range) == 2:
                if year_range[0] != year_range[1]:
                    year_filter = f"{year_range[0]}-{year_range[1]}"
                else:
                    year_filter = str(year_range[0])
        
        # Perform search
        if search_query and (search_query != st.session_state.last_search_query or search_clicked):
            if not st.session_state.rate_limit_time or (st.session_state.rate_limit_time and (st.session_state.rate_limit_time - datetime.datetime.now()).total_seconds() <= 0):
                with st.spinner("🔍 Searching Semantic Scholar..."):
                    papers = search_semantic_scholar(search_query, limit=20, year_filter=year_filter)
                    if papers:
                        st.session_state.cached_papers[search_query.lower()] = papers
                        st.session_state.last_search_query = search_query
                        
                        # Rank papers (clustering removed for Semantic Scholar due to issues)
                        with st.spinner("📊 Ranking papers by relevance..."):
                            st.session_state.ranked_papers = rank_papers_by_relevance(papers, search_query)
                        # Clear clusters for Semantic Scholar
                        st.session_state.clusters = {}
            else:
                # Use cached if available
                if search_query.lower() in st.session_state.cached_papers:
                    papers = st.session_state.cached_papers[search_query.lower()]
                    st.info(f"📦 Using cached results ({len(papers)} papers)")
        elif search_query and search_query.lower() in st.session_state.cached_papers:
            papers = st.session_state.cached_papers[search_query.lower()]
    
    # Note: Filters (include/exclude keywords, scope) have been removed per user request
    
    # Store papers in session state for feedback tracking and paper details access
    if papers:
        # Store current papers in session state so paper details can access them
        st.session_state.current_papers = papers
        
        # Update all_loaded_papers with current papers (avoid duplicates)
        seen_ids = {p.get('id', hash(p.get('title', ''))) for p in st.session_state.all_loaded_papers}
        for p in papers:
            paper_id = p.get('id', hash(p.get('title', '')))
            if paper_id not in seen_ids:
                st.session_state.all_loaded_papers.append(p)
                seen_ids.add(paper_id)
    else:
        # Clear current papers if no papers found
        if 'current_papers' in st.session_state:
            del st.session_state.current_papers
    
    # Display results
    if papers:
        st.header(f"📄 {len(papers)} Papers Found")
        
        # For Local Papers: Show both Clusters and Review Queue
        # For Semantic Scholar: Show only Review Queue (clustering has issues)
        if data_source == "Local Papers":
            # Tabs for Clusters and Review Queue
            tab_clusters, tab_queue = st.tabs(["📚 Clusters", "📋 Review Queue"])
            
            with tab_clusters:
                if st.session_state.clusters:
                    for cluster_name, cluster_data in st.session_state.clusters.items():
                        cluster_papers_list = [papers[i] for i in cluster_data["papers"] if 0 <= i < len(papers)]
                        if cluster_papers_list:
                            with st.expander(f"**{cluster_name}** ({len(cluster_papers_list)} papers)", expanded=False):
                                if cluster_data.get("topics"):
                                    st.caption(f"Topics: {', '.join(cluster_data['topics'][:5])}")
                                
                                for paper in cluster_papers_list:
                                    paper_id = paper.get('id', hash(paper['title']))
                                    col_paper1, col_paper2 = st.columns([4, 1])
                                    
                                    with col_paper1:
                                        if st.button(f"📄 {paper['title'][:60]}...", key=f"select_{paper_id}_cluster", use_container_width=True):
                                            st.session_state.selected_paper_id = paper_id
                                            st.rerun()
                                    
                                    with col_paper2:
                                        if paper.get('citation_count'):
                                            st.caption(f"⭐ {paper['citation_count']}")
                                    
                                    journal_display = paper.get('journal', 'N/A')
                                    if paper.get('volume') or paper.get('issue'):
                                        journal_display += f" ({paper.get('year', 'N/A')})"
                                    else:
                                        journal_display += f" • {paper.get('year', 'N/A')}"
                                    st.caption(journal_display)
                                    st.divider()
                else:
                    st.info("Clustering in progress...")
            
            with tab_queue:
                ranked = st.session_state.ranked_papers if st.session_state.ranked_papers else papers
                
                st.caption("Papers ranked by relevance to your search")
                
                for idx, paper in enumerate(ranked, 1):
                    paper_id = paper.get('id', hash(paper['title']))
                    
                    # Paper card
                    paper_selected = st.session_state.selected_paper_id == paper_id
                    card_style = "border: 2px solid #1f77b4;" if paper_selected else ""
                    
                    with st.container():
                        col_q1, col_q2 = st.columns([4, 1])
                        
                        with col_q1:
                            st.markdown(f"**{idx}. {paper['title']}**")
                            st.caption(f"{', '.join(paper.get('authors', ['Unknown'])[:3])} • {paper.get('journal', 'N/A')} • {paper.get('year', 'N/A')}")
                            if paper.get('citation_count'):
                                st.caption(f"⭐ {paper['citation_count']} citations")
                            elif paper.get('keywords'):
                                st.caption(f"🏷️ {', '.join(paper['keywords'][:3])}")
                        
                        with col_q2:
                            if st.button("View", key=f"view_{paper_id}", use_container_width=True):
                                st.session_state.selected_paper_id = paper_id
                                st.rerun()
                        
                        st.divider()
        else:
            # Semantic Scholar: Only Review Queue (no clusters)
            ranked = st.session_state.ranked_papers if st.session_state.ranked_papers else papers
            
            st.caption("Papers ranked by relevance to your search")
            
            for idx, paper in enumerate(ranked, 1):
                paper_id = paper.get('id', hash(paper['title']))
                
                # Paper card
                paper_selected = st.session_state.selected_paper_id == paper_id
                card_style = "border: 2px solid #1f77b4;" if paper_selected else ""
                
                with st.container():
                    col_q1, col_q2 = st.columns([4, 1])
                    
                    with col_q1:
                        st.markdown(f"**{idx}. {paper['title']}**")
                        st.caption(f"{', '.join(paper.get('authors', ['Unknown'])[:3])} • {paper.get('journal', 'N/A')} • {paper.get('year', 'N/A')}")
                        if paper.get('citation_count'):
                            st.caption(f"⭐ {paper['citation_count']} citations")
                        elif paper.get('keywords'):
                            st.caption(f"🏷️ {', '.join(paper['keywords'][:3])}")
                    
                    with col_q2:
                        if st.button("View", key=f"view_{paper_id}", use_container_width=True):
                            st.session_state.selected_paper_id = paper_id
                            st.rerun()
                    
                    st.divider()
            ranked = st.session_state.ranked_papers if st.session_state.ranked_papers else papers
            
            st.caption("Papers ranked by relevance to your search")
            
            for idx, paper in enumerate(ranked, 1):
                paper_id = paper.get('id', hash(paper['title']))
                
                # Paper card
                paper_selected = st.session_state.selected_paper_id == paper_id
                card_style = "border: 2px solid #1f77b4;" if paper_selected else ""
                
                with st.container():
                    col_q1, col_q2 = st.columns([4, 1])
                    
                    with col_q1:
                        st.markdown(f"**{idx}. {paper['title']}**")
                        authors_display = ', '.join(paper.get('authors', ['Unknown'])[:3])
                        journal_display = paper.get('journal', 'N/A')
                        year_display = paper.get('year', 'N/A')
                        st.caption(f"{authors_display} • {journal_display} • {year_display}")
                        if paper.get('citation_count'):
                            st.caption(f"⭐ {paper['citation_count']} citations")
                        elif paper.get('keywords'):
                            st.caption(f"🏷️ {', '.join(paper['keywords'][:3])}")
                    
                    with col_q2:
                        if st.button("View", key=f"view_{paper_id}", use_container_width=True):
                            st.session_state.selected_paper_id = paper_id
                            st.rerun()
                    
                    st.divider()
    else:
        if data_source == "Local Papers":
            st.info("📚 **Local Papers Mode**\n\nAll papers from `papers.json` are shown. Use the filter box to search within them.")
        else:
            st.info("👆 Enter a search query to find papers from Semantic Scholar")

# ==================== RIGHT COLUMN: SELECTED PAPER DETAILS ====================
with col_details:
    st.header("📖 Paper Details")
    
    # Find selected paper
    selected_paper = None
    all_papers = []
    
    # Get papers from current data source
    all_papers = []
    
    # First check ranked papers
    if st.session_state.ranked_papers:
        all_papers = st.session_state.ranked_papers
    # Then check current papers from session state
    elif st.session_state.get('current_papers'):
        all_papers = st.session_state.current_papers
    
    # Only show paper details if we have papers and a selected paper ID
    if st.session_state.selected_paper_id and all_papers:
        for p in all_papers:
            paper_id = p.get('id', hash(p['title']))
            if paper_id == st.session_state.selected_paper_id:
                selected_paper = p
                break
    
    if selected_paper:
        # Paper header
        st.markdown(f"### {selected_paper['title']}")
        
        # Authors
        authors_str = ', '.join(selected_paper.get('authors', ['Unknown'])[:5])
        st.markdown(f"**Authors:** {authors_str}")
        
        # Publication info
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            journal_info = selected_paper.get('journal', 'N/A')
            # Add volume/issue/pages if available (for local papers)
            if selected_paper.get('volume') or selected_paper.get('issue') or selected_paper.get('pages'):
                journal_info += f", Vol {selected_paper.get('volume', '')}({selected_paper.get('issue', '')}), pp. {selected_paper.get('pages', '')}"
            st.markdown(f"**Journal:** {journal_info}")
            st.markdown(f"**Year:** {selected_paper.get('year', 'N/A')}")
        with col_info2:
            if selected_paper.get('citation_count'):
                st.metric("Citations", selected_paper['citation_count'])
            if selected_paper.get('doi'):
                st.markdown(f"**DOI:** {selected_paper['doi']}")
        
        st.divider()
        
        # Abstract
        st.markdown("**Abstract:**")
        st.write(selected_paper.get('abstract', 'No abstract available'))
        
        # Keywords/Fields of Study (if available)
        if selected_paper.get('fieldsOfStudy'):
            st.markdown(f"**Fields of Study:** {', '.join(selected_paper['fieldsOfStudy'])}")
        elif selected_paper.get('keywords'):
            st.markdown(f"**Keywords:** {', '.join(selected_paper['keywords'])}")
        
        # URL
        if selected_paper.get('url'):
            st.markdown(f"[🔗 View on Semantic Scholar]({selected_paper['url']})")
        
        st.divider()
        
        # Action buttons
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("➕ Add to List", key=f"add_{selected_paper.get('id')}", use_container_width=True, type="primary"):
                if selected_paper not in st.session_state.selected_papers:
                    st.session_state.selected_papers.append(selected_paper)
                    st.success("Added!")
                    st.rerun()
        
        with col_btn2:
            if st.button("🤖 Explain Relevance", key=f"explain_{selected_paper.get('id')}", use_container_width=True):
                with st.spinner("Generating explanation..."):
                    try:
                        # Use appropriate query based on data source
                        query_for_explanation = ""
                        if data_source == "Semantic Scholar API":
                            query_for_explanation = st.session_state.get('last_search_query', '')
                        else:
                            query_for_explanation = st.session_state.get('local_search', '') or "research papers"
                        
                        explanation = explain_relevance(selected_paper, query_for_explanation)
                        st.session_state.ai_explanations[selected_paper.get('id')] = explanation
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        st.divider()
        
        # User Feedback Section
        st.subheader("💬 Feedback")
        
        paper_id = selected_paper.get('id')
        feedback_key = f"feedback_{paper_id}"
        
        if feedback_key not in st.session_state.paper_feedback:
            st.session_state.paper_feedback[feedback_key] = {"relevant": None, "note": ""}
        
        col_fb1, col_fb2 = st.columns(2)
        with col_fb1:
            if st.button("✅ Relevant", key=f"rel_{paper_id}", use_container_width=True):
                st.session_state.paper_feedback[feedback_key]["relevant"] = True
                st.success("Marked as relevant!")
        with col_fb2:
            if st.button("❌ Not Relevant", key=f"notrel_{paper_id}", use_container_width=True):
                st.session_state.paper_feedback[feedback_key]["relevant"] = False
                st.info("Marked as not relevant")
        
        # Show current feedback status
        if st.session_state.paper_feedback[feedback_key]["relevant"] is not None:
            status = "✅ Relevant" if st.session_state.paper_feedback[feedback_key]["relevant"] else "❌ Not Relevant"
            st.caption(f"Status: {status}")
        
        # Note field
        note = st.text_area("Quick note (optional)", key=f"note_{paper_id}", height=80)
        if st.button("💾 Save Note", key=f"save_note_{paper_id}"):
            st.session_state.paper_feedback[feedback_key]["note"] = note
            st.success("Note saved!")
        
        st.divider()
        
        # AI Explanation
        st.subheader("🤖 AI Explanation")
        
        if paper_id in st.session_state.ai_explanations:
            st.info(st.session_state.ai_explanations[paper_id])
        else:
            st.caption("Click 'Explain Relevance' button above to get AI-generated relevance explanation")
    
    else:
        if not all_papers:
            st.info("👈 Search for papers or select Local Papers to view details")
        else:
            st.info("👈 Select a paper from the results to view details")

# ==================== BOTTOM: READING LIST ====================
st.divider()
st.header("📌 Your Reading List")

if st.session_state.selected_papers:
    for idx, p in enumerate(st.session_state.selected_papers, 1):
        col_list1, col_list2 = st.columns([4, 1])
        with col_list1:
            st.markdown(f"{idx}. **{p['title']}** ({p.get('year', 'N/A')}) — {p.get('journal', 'N/A')}")
        with col_list2:
            if st.button("Remove", key=f"remove_{idx}"):
                st.session_state.selected_papers.remove(p)
                st.rerun()
else:
    st.caption("No papers in your reading list yet. Add papers using the 'Add to List' button.")

# ==================== FEEDBACK SUMMARY SECTION ====================
st.divider()
st.header("📊 Your Feedback Summary")

# Collect all papers with feedback
def get_papers_with_feedback():
    """Collect all papers that have been marked as relevant or not relevant"""
    relevant_papers = []
    not_relevant_papers = []
    
    # Use all loaded papers from session state
    all_available_papers = st.session_state.all_loaded_papers.copy()
    
    # Also add papers from cached searches
    for cached_papers_list in st.session_state.cached_papers.values():
        all_available_papers.extend(cached_papers_list)
    
    # Remove duplicates based on paper ID
    seen_ids = set()
    unique_papers = []
    for p in all_available_papers:
        paper_id = p.get('id', hash(p.get('title', '')))
        if paper_id not in seen_ids:
            seen_ids.add(paper_id)
            unique_papers.append(p)
    
    # Check feedback for each paper
    for paper in unique_papers:
        paper_id = paper.get('id', hash(paper.get('title', '')))
        feedback_key = f"feedback_{paper_id}"
        
        if feedback_key in st.session_state.paper_feedback:
            feedback = st.session_state.paper_feedback[feedback_key]
            if feedback.get("relevant") is True:
                relevant_papers.append({
                    "paper": paper,
                    "note": feedback.get("note", "")
                })
            elif feedback.get("relevant") is False:
                not_relevant_papers.append({
                    "paper": paper,
                    "note": feedback.get("note", "")
                })
    
    return relevant_papers, not_relevant_papers

# Get papers with feedback
relevant_papers, not_relevant_papers = get_papers_with_feedback()

if relevant_papers or not_relevant_papers:
    tab_relevant, tab_not_relevant = st.tabs([
        f"✅ Relevant ({len(relevant_papers)})",
        f"❌ Not Relevant ({len(not_relevant_papers)})"
    ])
    
    with tab_relevant:
        if relevant_papers:
            st.caption(f"You've marked {len(relevant_papers)} paper(s) as relevant to your research.")
            for idx, item in enumerate(relevant_papers, 1):
                paper = item["paper"]
                note = item["note"]
                paper_id = paper.get('id', hash(paper.get('title', '')))
                
                with st.expander(f"{idx}. **{paper.get('title', 'Untitled')}**", expanded=False):
                    col_fb1, col_fb2 = st.columns([4, 1])
                    with col_fb1:
                        st.markdown(f"**Authors:** {', '.join(paper.get('authors', ['Unknown'])[:3])}")
                        st.markdown(f"**Journal:** {paper.get('journal', 'N/A')} • **Year:** {paper.get('year', 'N/A')}")
                        if note:
                            st.info(f"📝 **Your Note:** {note}")
                    with col_fb2:
                        if st.button("View", key=f"view_fb_rel_{paper_id}", use_container_width=True):
                            st.session_state.selected_paper_id = paper_id
                            st.rerun()
        else:
            st.info("No papers marked as relevant yet. Use the '✅ Relevant' button on paper details to mark papers.")
    
    with tab_not_relevant:
        if not_relevant_papers:
            st.caption(f"You've marked {len(not_relevant_papers)} paper(s) as not relevant.")
            for idx, item in enumerate(not_relevant_papers, 1):
                paper = item["paper"]
                note = item["note"]
                paper_id = paper.get('id', hash(paper.get('title', '')))
                
                with st.expander(f"{idx}. **{paper.get('title', 'Untitled')}**", expanded=False):
                    col_fb1, col_fb2 = st.columns([4, 1])
                    with col_fb1:
                        st.markdown(f"**Authors:** {', '.join(paper.get('authors', ['Unknown'])[:3])}")
                        st.markdown(f"**Journal:** {paper.get('journal', 'N/A')} • **Year:** {paper.get('year', 'N/A')}")
                        if note:
                            st.info(f"📝 **Your Note:** {note}")
                    with col_fb2:
                        if st.button("View", key=f"view_fb_notrel_{paper_id}", use_container_width=True):
                            st.session_state.selected_paper_id = paper_id
                            st.rerun()
        else:
            st.info("No papers marked as not relevant yet. Use the '❌ Not Relevant' button on paper details to mark papers.")
else:
    st.info("💡 **No feedback yet**\n\nMark papers as relevant or not relevant using the feedback buttons in the paper details panel to see them organized here.")
