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
# Helper Functions
# -----------------------------
def get_consistent_paper_id(paper):
    """
    Return a stable, consistent paper ID for all papers.
    Works for:
    - OpenAlex papers (paperId like 'W123456789')
    - Local papers (numeric id)
    - Fallback (hash of title + year)
    """
    if paper.get("paperId") and isinstance(paper["paperId"], str):
        return str(paper["paperId"])

    if paper.get("id") is not None:
        return str(paper["id"])

    return str(hash(f"{paper.get('title','')}_{paper.get('year','')}"))

# -----------------------------
# OpenAlex API functions
# -----------------------------
def search_openalex(query: str, limit: int = 20, year_filter: str = None, min_citations: int = None):
    """
    Search OpenAlex API for papers using the works endpoint
    Documentation: https://docs.openalex.org/api-entities/works/search-works
    OpenAlex has generous rate limits (10 requests/second) - no rate limit handling needed
    """
    query = query.strip()
    if not query:
        return []
    
    # Clean query - remove redundant words (but keep the query meaningful)
    query = query.replace(" papers", "").replace(" paper", "")
    query = query.replace(" articles", "").replace(" article", "")
    query = query.strip()
    
    # Ensure query is still valid after cleaning
    if not query or len(query) < 2:
        st.warning("⚠️ Search query is too short. Please enter at least 2 characters.")
        return []
    
    # Ensure limit is within valid range (max 200 per API docs, but we'll use 25 for performance)
    limit = max(1, min(limit, 25))
    
    # Use the OpenAlex works endpoint
    url = "https://api.openalex.org/works"
    
    # Build parameters according to OpenAlex API documentation
    # OpenAlex uses 'search' parameter for full-text search
    params = {
        "search": query,
        "per_page": limit
    }
    
    # Note: OpenAlex doesn't support 'select' parameter in the same way
    # We'll get all fields and extract what we need
    
    # Add optional filters
    filters = []
    
    if year_filter:
        # Parse year filter (format: "2016-2020" or "2019")
        if "-" in year_filter:
            years = year_filter.split("-")
            try:
                start_year = int(years[0])
                end_year = int(years[1])
                filters.append(f"publication_year:{start_year}-{end_year}")
            except:
                pass
        else:
            try:
                year = int(year_filter)
                filters.append(f"publication_year:{year}")
            except:
                pass
    
    if min_citations:
        filters.append(f"cited_by_count:>={min_citations}")
    
    # Combine filters with comma separation
    if filters:
        params["filter"] = ",".join(filters)
    
    try:
        # Make request with proper headers
        headers = {
            'User-Agent': 'LitSense/1.0 (mailto:user@example.com)'  # OpenAlex prefers identifying user agents
        }
        
        # Debug: Log the request URL (remove in production)
        # st.write(f"Debug: Requesting {url} with params: {params}")
        
        response = requests.get(url, params=params, headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            # OpenAlex returns: meta, results[]
            papers_data = data.get('results', [])
            
            # If no results, return empty list
            if not papers_data:
                return []
            
            papers = []
            for paper in papers_data:
                # Extract title
                title = paper.get('display_name') or paper.get('title', 'Untitled')
                
                # Extract authors from authorships array
                authors = []
                if paper.get('authorships'):
                    for authorship in paper.get('authorships', []):
                        author = authorship.get('author')
                        if author:
                            author_name = author.get('display_name', '')
                            if author_name:
                                authors.append(author_name)
                
                # Extract year
                year = paper.get('publication_year') or 0
                
                # Extract abstract
                abstract = paper.get('abstract', 'No abstract available')
                # OpenAlex abstracts are sometimes in inverted format, check if it starts with inverted
                if abstract.startswith("InvertedAbstract"):
                    # Try to extract the actual abstract
                    abstract = abstract.replace("InvertedAbstract", "").strip()
                
                # Extract journal/venue from primary_location
                journal = "N/A"
                if paper.get('primary_location'):
                    source = paper.get('primary_location', {}).get('source')
                    if source:
                        journal = source.get('display_name', 'N/A')
                
                # Extract DOI
                doi = paper.get('doi', '').replace('https://doi.org/', '') if paper.get('doi') else ''
                
                # Extract concepts (similar to fields of study)
                concepts = []
                if paper.get('concepts'):
                    concepts = [concept.get('display_name', '') for concept in paper.get('concepts', [])[:5] if concept.get('display_name')]
                
                # Extract citation count
                citation_count = paper.get('cited_by_count', 0)
                
                # Extract OpenAlex ID (URL format: https://openalex.org/W123456789)
                openalex_id = paper.get('id', '').replace('https://openalex.org/', '') if paper.get('id') else ''
                
                # Extract URL
                url_link = paper.get('primary_location', {}).get('landing_page_url', '') or paper.get('id', '')
                
                # Use OpenAlex ID as primary identifier, or create unique hash
                unique_id = openalex_id or hash(f"{title}_{year}")
                
                paper_obj = {
                    "id": unique_id,
                    "paperId": openalex_id,  # Store OpenAlex ID
                    "title": title,
                    "authors": authors if authors else ["Unknown"],
                    "year": year,
                    "abstract": abstract if abstract else "No abstract available",
                    "journal": journal,
                    "doi": doi,
                    "keywords": concepts,  # Use concepts as keywords
                    "citation_count": citation_count,
                    "url": url_link,
                    "fieldsOfStudy": concepts,
                    "publicationTypes": []
                }
                papers.append(paper_obj)
            
            return papers
        
        elif response.status_code == 429:
            # OpenAlex rate limit (very rare, but handle it)
            st.warning("⏰ OpenAlex rate limit reached. Please wait a moment and try again.")
            return []
        
        elif response.status_code == 400:
            # Try to get more details from the error response
            try:
                error_data = response.json()
                error_msg = error_data.get('message', 'Invalid search query')
                st.error(f"❌ OpenAlex API error: {error_msg}")
                st.info(f"💡 **Troubleshooting:**\n- Make sure your search query is not empty\n- Try simpler search terms\n- Check if special characters are causing issues")
            except:
                # If we can't parse JSON, show the raw response text
                try:
                    error_text = response.text[:200]  # First 200 chars
                    st.error(f"❌ Invalid search query. API response: {error_text}")
                except:
                    st.error("❌ Invalid search query. Try a different search term.")
            return []
        
        elif response.status_code == 429:
            # OpenAlex rate limit (very rare, but handle it)
            st.warning("⏰ OpenAlex rate limit reached. Please wait a moment and try again.")
            return []
        
        else:
            # Try to get error details
            try:
                error_data = response.json()
                error_msg = error_data.get('message', f'API error: {response.status_code}')
                st.error(f"❌ OpenAlex API error ({response.status_code}): {error_msg}")
            except:
                st.error(f"❌ OpenAlex API error: {response.status_code}")
            return []
            
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Network error connecting to OpenAlex: {str(e)}")
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
# AI helper - Summarize paper
# -----------------------------
def summarize_paper(paper):
    """Generate a concise summary of a paper using Gemini"""
    if not paper or not GEMINI_API_KEY:
        return "Summary not available"
    
    try:
        authors_str = ", ".join(paper.get('authors', ['Unknown'])[:3])
        
        prompt = f"""Provide a concise 2-3 sentence summary of this research paper.

Title: {paper.get('title', 'N/A')}
Authors: {authors_str}
Year: {paper.get('year', 'N/A')}
Journal: {paper.get('journal', 'N/A')}

Abstract:
{paper.get('abstract', 'No abstract available')}

Focus on:
1. What problem/question does this paper address?
2. What is the main contribution or finding?
3. Why is this important?

Keep it brief and informative."""
        
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating summary: {str(e)}"

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
        # Add any papers not in ranking (avoid duplicates)
        ranked_indices_set = set(ranked_indices)
        remaining = [p for i, p in enumerate(papers) if i not in ranked_indices_set]
        return ranked + remaining
    except:
        return papers

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="LitSense", layout="wide", initial_sidebar_state="collapsed")

# Initialize session state
if "selected_papers" not in st.session_state:
    st.session_state.selected_papers = []
if "ai_explanations" not in st.session_state:
    st.session_state.ai_explanations = {}
if "paper_summaries" not in st.session_state:
    st.session_state.paper_summaries = {}  # Cache for paper summaries
if "scroll_to_section" not in st.session_state:
    st.session_state.scroll_to_section = None  # Track which section to scroll to
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
if "study_mode" not in st.session_state:
    st.session_state.study_mode = None  # "ai" or "baseline"
if "study_condition" not in st.session_state:
    # Randomly assign condition: 0 = AI first, 1 = Baseline first
    st.session_state.study_condition = random.choice([0, 1])
if "task_completed" not in st.session_state:
    st.session_state.task_completed = False
if "survey_completed" not in st.session_state:
    st.session_state.survey_completed = False
if "show_instructions" not in st.session_state:
    st.session_state.show_instructions = True
if "completed_modes" not in st.session_state:
    st.session_state.completed_modes = []  # Track which modes have been completed
if "first_survey_submitted" not in st.session_state:
    st.session_state.first_survey_submitted = False  # Track if first survey was submitted

# Study Mode Selection (only show if not set)
if st.session_state.study_mode is None:
    st.title("📚 Literature Review Study")
    st.markdown("---")
    
    st.info("""
    **Welcome to the Literature Review Study!**
    
    This study compares two different interfaces for literature review. You will be randomly assigned to use one interface first, then switch to the other.
    
    **What you'll do:**
    1. Complete a literature review task using the assigned interface
    2. Switch to the other interface and complete the same task
    3. Complete a brief survey about your experience
    
    Your participation is anonymous and voluntary.
    """)
    
    col_mode1, col_mode2 = st.columns(2)
    
    with col_mode1:
        if st.button("🤖 Start with AI Mode", use_container_width=True, type="primary"):
            st.session_state.study_mode = "ai"
            st.session_state.show_instructions = True
            st.rerun()
    
    with col_mode2:
        if st.button("📋 Start with Baseline Mode", use_container_width=True, type="primary"):
            st.session_state.study_mode = "baseline"
            st.session_state.show_instructions = True
            st.rerun()
    
    st.stop()

# If both modes completed but Finish not clicked yet, don't show main interface
# The survey section will handle showing the Finish button

# Show task instructions if needed (separate step before main interface)
if st.session_state.show_instructions and not st.session_state.task_completed:
    st.title("📚 Literature Review Study")
    st.markdown("---")
    
    mode_badge = "🤖 **AI Mode**" if st.session_state.study_mode == "ai" else "📋 **Baseline Mode**"
    st.info(f"""
    **Current Mode:** {mode_badge}
    
    **Your Task:**
    
    Please use this interface to find and review papers related to your research topic. 
    
    **Steps:**
    1. Search for papers using the search interface (left column)
    2. Browse through the results (middle column)
    3. Click on papers to view details (right column)
    4. Add relevant papers to your reading list
    5. Mark papers as relevant or not relevant based on your research needs
    
    **Note:** After completing your task, you'll be asked to complete a brief survey about your experience.
    """)
    
    if st.button("✅ I understand, start the task", use_container_width=True, type="primary"):
        st.session_state.show_instructions = False
        st.rerun()
    
    st.stop()  # Stop here until user clicks the button

# Show main interface only if task is not completed and instructions are dismissed
if not st.session_state.task_completed and not st.session_state.show_instructions:
    # Header
    col_header1, col_header2, col_header3 = st.columns([3, 1, 1])
    with col_header1:
        st.title("📚 Literature Review Assistant")
        st.caption("Search → Explore → Review → Refine")
        # Show current mode badge
        mode_badge = "🤖 AI Mode" if st.session_state.study_mode == "ai" else "📋 Baseline Mode"
        st.caption(f"**Current Mode:** {mode_badge}")
    with col_header2:
        if st.session_state.selected_papers:
            st.metric("Saved", len(st.session_state.selected_papers))
    with col_header3:
        if st.button("✅ Complete Task", use_container_width=True, type="primary"):
            st.session_state.task_completed = True
            st.rerun()

    # Main layout: 3 columns
    col_search, col_results, col_details = st.columns([1, 2, 1.5])

    # ==================== LEFT COLUMN: SEARCH & STEERING ====================
    with col_search:
        st.header("🔍 Search & Filter")
    
    # Data source selection
    data_source = st.radio(
        "Data Source",
        ["Search Online", "Local Papers"],
        help="Choose between live online search or local papers",
        key="data_source_selector"
    )
    
    # Note about loading time (only show AI note in AI mode)
    if st.session_state.study_mode == "ai":
        if data_source == "Search Online":
            st.caption("⏱️ Note: Processing may take 10-15 seconds (AI clustering & ranking)")
        else:
            st.caption("⏱️ Note: Processing may take 10-15 seconds (AI clustering & ranking)")
    else:
        st.caption("⏱️ Note: Processing may take a few seconds")
    
    # Reset state when switching data sources
    if data_source != st.session_state.get('last_data_source'):
        if st.session_state.get('last_data_source') is not None:
            st.session_state.selected_paper_id = None
            st.session_state.ai_explanations = {}
            st.session_state.paper_summaries = {}
            st.session_state.current_papers = []
            # Clear AI-generated data when switching data sources
            # (will be regenerated in AI mode if needed)
            st.session_state.clusters = {}
            st.session_state.ranked_papers = []
            # Clear rate limit timer when switching
            st.session_state.rate_limit_time = None
        st.session_state.last_data_source = data_source
    
    # Ensure baseline mode never has AI-generated data
    if st.session_state.study_mode == "baseline":
        # Clear any AI-generated data that might exist
        if st.session_state.clusters:
            st.session_state.clusters = {}
        # In baseline mode, ranked_papers should be same as original papers
        # This is handled in the search logic above, but ensure it's cleared if no papers
        if not st.session_state.get('current_papers') and st.session_state.ranked_papers:
            # Only clear if we're not actively displaying papers
            pass  # Keep ranked_papers for display consistency, but it should be set to papers in baseline mode
    
    st.divider()
    
    if data_source == "Search Online":
        search_query = st.text_input(
            "Research Topic",
            placeholder="e.g. machine learning, transformer architectures",
            help="Enter your research topic to search online",
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
                help="Filter papers by title, abstract, or keywords",
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
    
    # OpenAlex has generous rate limits (10 req/sec) - no rate limit warnings needed
    
    st.divider()
    
    # Year filter (only for Search Online)
    if data_source == "Search Online":
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
    
    # Cached searches (only for Search Online)
    if data_source == "Search Online" and st.session_state.cached_papers:
        st.subheader("📦 Recent Searches")
        for cached_query in list(st.session_state.cached_papers.keys())[:5]:
            if st.button(f"📄 {cached_query[:30]}...", key=f"cache_{cached_query}", use_container_width=True):
                st.session_state.last_search_query = cached_query
                st.rerun()
    
    # Show local papers info
    if data_source == "Local Papers":
        st.info("💡 **Local Papers Mode**\n\nShowing local papers. Use the filter box above to search within them.")

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
                st.error("❌ Local papers file not found!")
                return []
            except json.JSONDecodeError as e:
                st.error(f"❌ Invalid JSON in local papers file: {str(e)}")
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
        
        # Generate clusters and ranking for local papers (only if papers loaded and in AI mode)
        if papers:
            # Regenerate clusters/ranking when switching to local papers or on new search (only in AI mode)
            local_search_value = st.session_state.get('local_search', '')
            if st.session_state.study_mode == "ai":
                if data_source != st.session_state.get('last_data_source') or search_clicked or not st.session_state.clusters:
                    with st.spinner("🤖 Organizing papers into clusters..."):
                        cluster_query = local_search_value if local_search_value else "research papers"
                        st.session_state.clusters = cluster_papers(papers, cluster_query)
                    
                    with st.spinner("📊 Ranking papers by relevance..."):
                        rank_query = local_search_value if local_search_value else "general research"
                        st.session_state.ranked_papers = rank_papers_by_relevance(papers, rank_query)
            else:
                # Baseline mode: no AI ranking, use original order
                st.session_state.ranked_papers = papers
                st.session_state.clusters = {}
            
            st.session_state.last_data_source = data_source
    
    else:
        # Search Online mode
        # Get year filter if set
        year_filter = None
        if "year_filter" in st.session_state:
            year_range = st.session_state.year_filter
            if isinstance(year_range, (list, tuple)) and len(year_range) == 2:
                if year_range[0] != year_range[1]:
                    year_filter = f"{year_range[0]}-{year_range[1]}"
                else:
                    year_filter = str(year_range[0])
        
        # Perform search - OpenAlex has generous rate limits, no need for rate limit checks
        papers = []
        if search_query and (search_query != st.session_state.last_search_query or search_clicked):
            with st.spinner("🔍 Searching online..."):
                papers = search_openalex(search_query, limit=20, year_filter=year_filter)
                if papers:
                    st.session_state.cached_papers[search_query.lower()] = papers
                    st.session_state.last_search_query = search_query
                    
                    # Rank papers only in AI mode (clustering removed for online search due to issues)
                    if st.session_state.study_mode == "ai":
                        with st.spinner("📊 Ranking papers by relevance..."):
                            ranked_result = rank_papers_by_relevance(papers, search_query)
                            # Remove duplicates from ranked results
                            seen_ids = set()
                            unique_ranked = []
                            for p in ranked_result:
                                pid = get_consistent_paper_id(p)
                                if pid not in seen_ids:
                                    seen_ids.add(pid)
                                    unique_ranked.append(p)
                            st.session_state.ranked_papers = unique_ranked
                    else:
                        # Baseline mode: use original order
                        st.session_state.ranked_papers = papers
                    # Clear clusters for online search
                    st.session_state.clusters = {}
        elif search_query and search_query.lower() in st.session_state.cached_papers:
            papers = st.session_state.cached_papers[search_query.lower()]
            # In baseline mode, always use original papers (no AI ranking)
            if st.session_state.study_mode == "baseline":
                st.session_state.ranked_papers = papers
                st.session_state.clusters = {}
    
    # Note: Filters (include/exclude keywords, scope) have been removed per user request
    
    # Store papers in session state for feedback tracking and paper details access
    if papers:
        # Store current papers in session state so paper details can access them
        st.session_state.current_papers = papers
        
        # Update all_loaded_papers with current papers (avoid duplicates)
        # Use paperId or create unique hash for comparison
        seen_ids = set()
        for p in st.session_state.all_loaded_papers:
            pid = p.get('paperId') or p.get('id') or hash(f"{p.get('title', '')}_{p.get('year', '')}")
            seen_ids.add(pid)
        
        for p in papers:
            paper_id = p.get('paperId') or p.get('id') or hash(f"{p.get('title', '')}_{p.get('year', '')}")
            if paper_id not in seen_ids:
                st.session_state.all_loaded_papers.append(p)
                seen_ids.add(paper_id)
    else:
        # Clear current papers if no papers found
        if 'current_papers' in st.session_state:
            del st.session_state.current_papers
    
    # Display results (inside column)
    with col_results:
        # Ensure papers list doesn't have duplicates before displaying
        if papers:
            # Remove duplicates from papers list
            seen_paper_ids = set()
            unique_papers = []
            for p in papers:
                paper_id = get_consistent_paper_id(p)
                if paper_id not in seen_paper_ids:
                    seen_paper_ids.add(paper_id)
                    unique_papers.append(p)
            papers = unique_papers
            
            st.header(f"📄 {len(papers)} Papers Found")
            
            # For Local Papers: Show Clusters (only in AI mode), Review Queue, and Reading List
            # For Search Online: Show Review Queue and Reading List
            if data_source == "Local Papers":
                # Tabs for Clusters (AI mode only), Review Queue, and Reading List
                if st.session_state.study_mode == "ai":
                    tab_clusters, tab_queue, tab_reading = st.tabs(["📚 Clusters", "📋 Review Queue", "📖 Reading List"])
                else:
                    tab_queue, tab_reading = st.tabs(["📋 Review Queue", "📖 Reading List"])
                    tab_clusters = None
                
                if tab_clusters:
                    with tab_clusters:
                        if st.session_state.clusters:
                            for cluster_name, cluster_data in st.session_state.clusters.items():
                                cluster_papers_list = [papers[i] for i in cluster_data["papers"] if 0 <= i < len(papers)]
                                if cluster_papers_list:
                                    with st.expander(f"**{cluster_name}** ({len(cluster_papers_list)} papers)", expanded=False):
                                        if cluster_data.get("topics"):
                                            st.caption(f"Topics: {', '.join(cluster_data['topics'][:5])}")
                                        
                                        for cluster_idx, paper in enumerate(cluster_papers_list):
                                            # Get paper ID using consistent function
                                            paper_id = get_consistent_paper_id(paper)
                                            # Create unique key using cluster name and index
                                            unique_cluster_key = f"select_{cluster_name}_{cluster_idx}_{paper_id}"
                                            col_paper1, col_paper2 = st.columns([4, 1])
                                            
                                            with col_paper1:
                                                if st.button(f"📄 {paper['title'][:60]}...", key=unique_cluster_key, use_container_width=True):
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
                    if st.session_state.study_mode == "ai":
                        ranked = st.session_state.ranked_papers if st.session_state.ranked_papers else papers
                        st.caption("Papers ranked by relevance to your search")
                    else:
                        ranked = papers
                        st.caption("Papers (in search result order)")
                    
                    for idx, paper in enumerate(ranked, 1):
                        # Get paper ID using consistent function
                        paper_id = get_consistent_paper_id(paper)
                        # Create unique key using index to avoid duplicates
                        unique_key = f"view_queue_{idx}_{paper_id}"
                        
                        # Paper card
                        paper_selected = st.session_state.selected_paper_id == paper_id
                        card_style = "border: 2px solid #1f77b4;" if paper_selected else ""
                        
                        with st.container():
                            col_q1, col_q2 = st.columns([4, 1])
                            
                            with col_q1:
                                # Make title clickable (like local papers)
                                if st.button(f"📄 {idx}. {paper['title'][:70]}...", key=unique_key, use_container_width=True):
                                    st.session_state.selected_paper_id = paper_id
                                    st.rerun()
                                st.caption(f"{', '.join(paper.get('authors', ['Unknown'])[:3])} • {paper.get('journal', 'N/A')} • {paper.get('year', 'N/A')}")
                                if paper.get('citation_count'):
                                    st.caption(f"⭐ {paper['citation_count']} citations")
                                elif paper.get('keywords'):
                                    st.caption(f"🏷️ {', '.join(paper['keywords'][:3])}")
                            
                            with col_q2:
                                # Show citation count or other info in right column
                                if paper.get('citation_count'):
                                    st.caption(f"⭐ {paper['citation_count']}")
                            
                            st.divider()
                
                # Reading List tab for Local Papers
                with tab_reading:
                    if st.session_state.selected_papers:
                        st.caption(f"You have {len(st.session_state.selected_papers)} paper(s) in your reading list.")
                        for idx, paper in enumerate(st.session_state.selected_papers, 1):
                            paper_id = get_consistent_paper_id(paper)
                            unique_reading_key = f"reading_local_{idx}_{paper_id}"
                            
                            col_r1, col_r2 = st.columns([4, 1])
                            
                            with col_r1:
                                if st.button(f"📄 {idx}. {paper['title'][:70]}...", key=unique_reading_key, use_container_width=True):
                                    st.session_state.selected_paper_id = paper_id
                                    st.rerun()
                                st.caption(f"{', '.join(paper.get('authors', ['Unknown'])[:3])} • {paper.get('journal', 'N/A')} • {paper.get('year', 'N/A')}")
                            
                            with col_r2:
                                if st.button("🗑️ Remove", key=f"remove_local_{paper_id}", use_container_width=True):
                                    st.session_state.selected_papers = [p for p in st.session_state.selected_papers if get_consistent_paper_id(p) != paper_id]
                                    st.rerun()
                            
                            st.divider()
                    else:
                        st.info("📖 Your reading list is empty. Add papers using the '➕ Add to List' button in paper details.")
            else:
                # Search Online: Review Queue and Reading List (no clusters)
                tab_queue, tab_reading = st.tabs(["📋 Review Queue", "📖 Reading List"])
                
                # Review Queue tab content
                with tab_queue:
                    # In AI mode: use ranked papers if available, otherwise use papers
                    # In Baseline mode: always use original papers (no AI ranking)
                    if st.session_state.study_mode == "ai":
                        if st.session_state.ranked_papers:
                            ranked = st.session_state.ranked_papers
                        else:
                            ranked = papers
                        st.caption("Papers ranked by relevance to your search")
                    else:
                        # Baseline mode: use original order, ignore any existing ranked_papers
                        ranked = papers
                        st.caption("Papers (in search result order)")
                    
                    # Remove duplicates based on paper ID
                    seen_paper_ids = set()
                    unique_ranked = []
                    for paper in ranked:
                        paper_id = get_consistent_paper_id(paper)
                        if paper_id not in seen_paper_ids:
                            seen_paper_ids.add(paper_id)
                            unique_ranked.append(paper)
                    
                    for idx, paper in enumerate(unique_ranked, 1):
                        # Get paper ID using consistent function
                        paper_id = get_consistent_paper_id(paper)
                        # Create unique key using index to avoid duplicates
                        unique_key = f"view_scholar_{idx}_{paper_id}"
                        
                        # Paper card - clickable title like in local papers
                        paper_selected = st.session_state.selected_paper_id == paper_id
                        
                        with st.container():
                            col_q1, col_q2 = st.columns([4, 1])
                            
                            with col_q1:
                                # Make title clickable (like local papers)
                                if st.button(f"📄 {idx}. {paper['title'][:70]}...", key=unique_key, use_container_width=True):
                                    st.session_state.selected_paper_id = paper_id
                                    st.rerun()
                                st.caption(f"{', '.join(paper.get('authors', ['Unknown'])[:3])} • {paper.get('journal', 'N/A')} • {paper.get('year', 'N/A')}")
                                if paper.get('citation_count'):
                                    st.caption(f"⭐ {paper['citation_count']} citations")
                                elif paper.get('keywords'):
                                    st.caption(f"🏷️ {', '.join(paper['keywords'][:3])}")
                            
                            with col_q2:
                                # Show citation count or other info in right column
                                if paper.get('citation_count'):
                                    st.caption(f"⭐ {paper['citation_count']}")
                            
                            st.divider()
                
                # Reading List tab for Search Online
                with tab_reading:
                    if st.session_state.selected_papers:
                        st.caption(f"You have {len(st.session_state.selected_papers)} paper(s) in your reading list.")
                        for idx, paper in enumerate(st.session_state.selected_papers, 1):
                            paper_id = get_consistent_paper_id(paper)
                            unique_reading_key = f"reading_online_{idx}_{paper_id}"
                            
                            col_r1, col_r2 = st.columns([4, 1])
                            
                            with col_r1:
                                if st.button(f"📄 {idx}. {paper['title'][:70]}...", key=unique_reading_key, use_container_width=True):
                                    st.session_state.selected_paper_id = paper_id
                                    st.rerun()
                                st.caption(f"{', '.join(paper.get('authors', ['Unknown'])[:3])} • {paper.get('journal', 'N/A')} • {paper.get('year', 'N/A')}")
                            
                            with col_r2:
                                if st.button("🗑️ Remove", key=f"remove_online_{paper_id}", use_container_width=True):
                                    st.session_state.selected_papers = [p for p in st.session_state.selected_papers if get_consistent_paper_id(p) != paper_id]
                                    st.rerun()
                            
                            st.divider()
                    else:
                        st.info("📖 Your reading list is empty. Add papers using the '➕ Add to List' button in paper details.")
        else:
            # No papers found
            if data_source == "Local Papers":
                st.info("📚 **Local Papers Mode**\n\nAll local papers are shown. Use the filter box to search within them.")
            else:
                st.info("👆 Enter a search query to find papers online")

    # ==================== RIGHT COLUMN: SELECTED PAPER DETAILS ====================
    with col_details:
        st.header("📖 Paper Details")
        
        # Find selected paper
        selected_paper = None
        all_papers = []
        
        # Get papers from current data source only
        # Only show paper details if the selected paper belongs to the current data source
        all_papers = []
        
        # Get papers based on current data source
        if data_source == "Local Papers":
            # For local papers, check current_papers first (most recent)
            if st.session_state.get('current_papers'):
                # Filter to only local papers (those without OpenAlex paperId starting with 'W')
                all_papers = [p for p in st.session_state.current_papers 
                             if not p.get('paperId') or not (isinstance(p.get('paperId'), str) and p.get('paperId').startswith('W'))]
            elif st.session_state.get('all_loaded_papers'):
                # Filter to only local papers (those without OpenAlex paperId format)
                all_papers = [p for p in st.session_state.all_loaded_papers 
                             if not p.get('paperId') or not (isinstance(p.get('paperId'), str) and p.get('paperId').startswith('W'))]
        else:
            # For Search Online, get papers based on mode
            # In AI mode: prioritize ranked_papers, then current_papers
            # In Baseline mode: use current_papers (no AI ranking)
            if st.session_state.study_mode == "ai":
                if st.session_state.ranked_papers:
                    # Use ranked papers (these are definitely from Search Online)
                    all_papers = st.session_state.ranked_papers
                elif st.session_state.get('current_papers'):
                    # Check if current_papers contains online papers (have paperId starting with 'W')
                    online_papers = [p for p in st.session_state.current_papers 
                                   if p.get('paperId') and isinstance(p.get('paperId'), str) and p.get('paperId').startswith('W')]
                    if online_papers:
                        all_papers = online_papers
                    else:
                        # If no online papers in current_papers, try all_loaded_papers
                        if st.session_state.get('all_loaded_papers'):
                            all_papers = [p for p in st.session_state.all_loaded_papers 
                                         if p.get('paperId') and isinstance(p.get('paperId'), str) and p.get('paperId').startswith('W')]
                        else:
                            all_papers = []
                else:
                    all_papers = []
            else:
                # Baseline mode: use current_papers directly (no AI ranking)
                if st.session_state.get('current_papers'):
                    # Check if current_papers contains online papers (have paperId starting with 'W')
                    online_papers = [p for p in st.session_state.current_papers 
                                   if p.get('paperId') and isinstance(p.get('paperId'), str) and p.get('paperId').startswith('W')]
                    if online_papers:
                        all_papers = online_papers
                    else:
                        # If no online papers in current_papers, try all_loaded_papers
                        if st.session_state.get('all_loaded_papers'):
                            all_papers = [p for p in st.session_state.all_loaded_papers 
                                         if p.get('paperId') and isinstance(p.get('paperId'), str) and p.get('paperId').startswith('W')]
                        else:
                            all_papers = []
                else:
                    all_papers = []
        
        # Fix Paper Selection Matching Logic
        selected_paper = None
        if st.session_state.selected_paper_id and all_papers:
            selected_id = str(st.session_state.selected_paper_id)
            
            for p in all_papers:
                if get_consistent_paper_id(p) == selected_id:
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
                st.markdown(f"[🔗 View Paper]({selected_paper['url']})")
            
            st.divider()
            
            # Get paper ID consistently using the helper function
            paper_id = get_consistent_paper_id(selected_paper)
            
            # Action buttons (AI features only in AI mode)
            if st.session_state.study_mode == "ai":
                col_btn1, col_btn2, col_btn3 = st.columns(3)
            else:
                col_btn1, = st.columns(1)
            
            with col_btn1:
                if st.button("➕ Add to List", key=f"add_{paper_id}", use_container_width=True, type="primary"):
                    if selected_paper not in st.session_state.selected_papers:
                        st.session_state.selected_papers.append(selected_paper)
                        st.success("Added!")
                        st.rerun()
            
            if st.session_state.study_mode == "ai":
                with col_btn2:
                    if st.button("📝 Summarize", key=f"summarize_{paper_id}", use_container_width=True):
                        with st.spinner("Generating summary..."):
                            try:
                                summary = summarize_paper(selected_paper)
                                st.session_state.paper_summaries[paper_id] = summary
                                st.session_state.scroll_to_section = "summary"  # Trigger scroll to summary
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                
                with col_btn3:
                    if st.button("🤖 Explain Relevance", key=f"explain_{paper_id}", use_container_width=True):
                        with st.spinner("Generating explanation..."):
                            try:
                                # Use appropriate query based on data source
                                query_for_explanation = ""
                                if data_source == "Search Online":
                                    query_for_explanation = st.session_state.get('last_search_query', '')
                                else:
                                    query_for_explanation = st.session_state.get('local_search', '') or "research papers"
                                
                                explanation = explain_relevance(selected_paper, query_for_explanation)
                                st.session_state.ai_explanations[paper_id] = explanation
                                st.session_state.scroll_to_section = "explanation"  # Trigger scroll to explanation
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
            
            st.divider()
            
            # AI Summary Section (only in AI mode, show if available)
            if st.session_state.study_mode == "ai" and paper_id in st.session_state.paper_summaries:
                # Add anchor for scrolling
                st.markdown(f'<div id="summary_{paper_id}"></div>', unsafe_allow_html=True)
                st.subheader("📝 AI Summary")
                st.info(st.session_state.paper_summaries[paper_id])
                st.divider()
            
            # AI Explanation Section (only in AI mode, show if available)
            if st.session_state.study_mode == "ai" and paper_id in st.session_state.ai_explanations:
                # Add anchor for scrolling
                st.markdown(f'<div id="explanation_{paper_id}"></div>', unsafe_allow_html=True)
                st.subheader("🤖 AI Explanation")
                st.info(st.session_state.ai_explanations[paper_id])
                st.divider()
            
            # Scroll script injection (after sections are rendered)
            if st.session_state.scroll_to_section:
                scroll_target = st.session_state.scroll_to_section
                target_id = f"{scroll_target}_{paper_id}"
                st.session_state.scroll_to_section = None  # Clear the trigger
                
                scroll_script = f"""
                <script>
                    (function() {{
                        setTimeout(function() {{
                            const element = document.getElementById('{target_id}');
                            if (element) {{
                                element.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                            }}
                        }}, 300);
                    }})();
                </script>
                """
                st.markdown(scroll_script, unsafe_allow_html=True)
            
            # User Feedback Section
            st.subheader("💬 Feedback")
            
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
        
        else:
            # Check if selected paper ID exists but doesn't belong to current data source
            if st.session_state.selected_paper_id and not selected_paper:
                # Clear the selected paper ID if it doesn't belong to current data source
                st.session_state.selected_paper_id = None
            
            if not all_papers:
                if data_source == "Local Papers":
                    st.info("👈 Load local papers to view details")
                else:
                    st.info("👈 Search for papers online to view details")
            else:
                st.info("👈 Select a paper from the results to view details")

# Show continue message after first survey is submitted (before second mode)
if st.session_state.get('first_survey_submitted') and len(st.session_state.completed_modes) == 1:
    st.divider()
    completed_mode_display = st.session_state.completed_modes[0]
    completed_display = "🤖 AI Mode" if completed_mode_display == "ai" else "📋 Baseline Mode"
    remaining_mode_display = "📋 Baseline Mode" if completed_mode_display == "ai" else "🤖 AI Mode"
    remaining_mode = "baseline" if completed_mode_display == "ai" else "ai"
    
    st.success(f"✅ You've completed the **{completed_display}**!")
    st.info(f"""
    **Next Step:**
    
    You need to complete the **{remaining_mode_display}** to finish the study.
    
    Click the button below to continue with the next mode.
    """)
    
    if st.button(f"🔄 Continue with {remaining_mode_display}", use_container_width=True, type="primary"):
        # Switch to the other mode
        st.session_state.study_mode = remaining_mode
        st.session_state.task_completed = False
        st.session_state.show_instructions = True
        st.session_state.survey_completed = False
        st.session_state.first_survey_submitted = False  # Reset flag
        # Clear state for the new mode
        st.session_state.ai_explanations = {}
        st.session_state.paper_summaries = {}
        st.session_state.clusters = {}
        st.session_state.ranked_papers = []
        st.session_state.selected_paper_id = None
        st.session_state.selected_papers = []
        st.session_state.paper_feedback = {}
        st.rerun()
    
    st.stop()  # Stop here - don't show survey questions or main interface

# ==================== TASK COMPLETION & SURVEY SECTION ====================
# Show survey when task is completed (hide main interface)
if st.session_state.task_completed and not st.session_state.survey_completed:
    st.divider()
    st.header("📋 Post-Task Survey")
    
    st.markdown("""
    Thank you for completing the task! Please answer the following questions about your experience.
    """)
    
    # Satisfaction questions
    st.subheader("Satisfaction")
    satisfaction = st.slider(
        "How satisfied were you with this interface?",
        min_value=1,
        max_value=5,
        value=3,
        help="1 = Very Dissatisfied, 5 = Very Satisfied",
        key=f"satisfaction_{st.session_state.study_mode}"
    )
    
    ease_of_use = st.slider(
        "How easy was it to use this interface?",
        min_value=1,
        max_value=5,
        value=3,
        help="1 = Very Difficult, 5 = Very Easy",
        key=f"ease_of_use_{st.session_state.study_mode}"
    )
    
    usefulness = st.slider(
        "How useful was this interface for finding relevant papers?",
        min_value=1,
        max_value=5,
        value=3,
        help="1 = Not Useful, 5 = Very Useful",
        key=f"usefulness_{st.session_state.study_mode}"
    )
    
    # Cognitive load questions
    st.subheader("Cognitive Load")
    mental_demand = st.slider(
        "How mentally demanding was the task?",
        min_value=1,
        max_value=5,
        value=3,
        help="1 = Very Low, 5 = Very High",
        key=f"mental_demand_{st.session_state.study_mode}"
    )
    
    effort = st.slider(
        "How much effort did you need to invest?",
        min_value=1,
        max_value=5,
        value=3,
        help="1 = Very Little, 5 = Very Much",
        key=f"effort_{st.session_state.study_mode}"
    )
    
    frustration = st.slider(
        "How frustrated did you feel?",
        min_value=1,
        max_value=5,
        value=3,
        help="1 = Not Frustrated, 5 = Very Frustrated",
        key=f"frustration_{st.session_state.study_mode}"
    )
    
    # Open-ended feedback
    st.subheader("Additional Feedback")
    feedback_text = st.text_area(
        "Please share any additional comments about your experience:",
        height=100,
        placeholder="Your feedback here...",
        key=f"feedback_{st.session_state.study_mode}"
    )
    
    # Check if this is the first or second mode
    st.subheader("Next Step")
    
    if len(st.session_state.completed_modes) == 0:
        # First mode - show submit button
        if st.button("✅ Submit Survey & Continue", use_container_width=True, type="primary"):
            # Store survey results
            survey_results = {
                "satisfaction": satisfaction,
                "ease_of_use": ease_of_use,
                "usefulness": usefulness,
                "mental_demand": mental_demand,
                "effort": effort,
                "frustration": frustration,
                "feedback": feedback_text,
                "mode": st.session_state.study_mode,
                "condition": st.session_state.study_condition
            }
            
            # Mark current mode as completed
            st.session_state.completed_modes.append(st.session_state.study_mode)
            
            # Store survey results
            if not hasattr(st.session_state, 'survey_results'):
                st.session_state.survey_results = []
            st.session_state.survey_results.append(survey_results)
            
            # Set flag to show continue message
            st.session_state.first_survey_submitted = True
            st.rerun()
    
    elif len(st.session_state.completed_modes) == 1:
        # Second mode - show Finish button (survey questions already shown above)
        # Store survey results when Finish is clicked
        if st.button("✅ Finish Study", use_container_width=True, type="primary"):
            # Store survey results
            survey_results = {
                "satisfaction": satisfaction,
                "ease_of_use": ease_of_use,
                "usefulness": usefulness,
                "mental_demand": mental_demand,
                "effort": effort,
                "frustration": frustration,
                "feedback": feedback_text,
                "mode": st.session_state.study_mode,
                "condition": st.session_state.study_condition
            }
            
            # Mark current mode as completed
            if st.session_state.study_mode not in st.session_state.completed_modes:
                st.session_state.completed_modes.append(st.session_state.study_mode)
            
            # Store survey results
            if not hasattr(st.session_state, 'survey_results'):
                st.session_state.survey_results = []
            st.session_state.survey_results.append(survey_results)
            
            # Finalize the study
            st.session_state.survey_completed = True
            st.rerun()

# Show continue message after first survey is submitted (before second mode)
if st.session_state.get('first_survey_submitted') and len(st.session_state.completed_modes) == 1:
    st.divider()
    completed_mode_display = st.session_state.completed_modes[0]
    completed_display = "🤖 AI Mode" if completed_mode_display == "ai" else "📋 Baseline Mode"
    remaining_mode_display = "📋 Baseline Mode" if completed_mode_display == "ai" else "🤖 AI Mode"
    remaining_mode = "baseline" if completed_mode_display == "ai" else "ai"
    
    st.success(f"✅ You've completed the **{completed_display}**!")
    st.info(f"""
    **Next Step:**
    
    You need to complete the **{remaining_mode_display}** to finish the study.
    
    Click the button below to continue with the next mode.
    """)
    
    if st.button(f"🔄 Continue with {remaining_mode_display}", use_container_width=True, type="primary"):
        # Switch to the other mode
        st.session_state.study_mode = remaining_mode
        st.session_state.task_completed = False
        st.session_state.show_instructions = True
        st.session_state.survey_completed = False
        st.session_state.first_survey_submitted = False  # Reset flag
        # Clear state for the new mode
        st.session_state.ai_explanations = {}
        st.session_state.paper_summaries = {}
        st.session_state.clusters = {}
        st.session_state.ranked_papers = []
        st.session_state.selected_paper_id = None
        st.session_state.selected_papers = []
        st.session_state.paper_feedback = {}
        st.rerun()
    
    st.stop()  # Stop here - don't show survey questions

# Show final thank you message only after Finish button is clicked
if st.session_state.survey_completed and len(st.session_state.completed_modes) >= 2:
    st.title("🎉 Thank You for Participating!")
    st.success("✅ **Study Completed**")
    st.markdown("""
    You have successfully completed both modes of the study:
    
    - ✅ **AI Mode** - Completed
    - ✅ **Baseline Mode** - Completed
    
    Your responses have been recorded. Thank you for your valuable feedback!
    
    The study is now complete. You may close this page.
    """)
    st.stop()  # Stop execution - study is fully complete

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
        paper_id = get_consistent_paper_id(p)
        if paper_id not in seen_ids:
            seen_ids.add(paper_id)
            unique_papers.append(p)
    
    # Check feedback for each paper
    for paper in unique_papers:
        paper_id = get_consistent_paper_id(paper)
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
                paper_id = get_consistent_paper_id(paper)
                
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
                paper_id = get_consistent_paper_id(paper)
                
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
