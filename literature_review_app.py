import streamlit as st
import json
import pandas as pd
from typing import List, Dict
import requests
from datetime import datetime
# Use the old API for now (more stable, widely available)
# The new google.genai API requires different setup
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")

st.title("Gemini Chat")

user_input = st.text_input("Enter your message:")
if user_input:
    response = model.generate_content(user_input)
    st.write(response.text)

# Page configuration
st.set_page_config(
    page_title="Literature Review Assistant",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'selected_articles' not in st.session_state:
    st.session_state.selected_articles = []
if 'articles_data' not in st.session_state:
    st.session_state.articles_data = []

def load_json_articles(file_path: str) -> List[Dict]:
    """Load articles from JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"File {file_path} not found!")
        return []

def search_semantic_scholar(query: str, api_key: str = None, limit: int = 10) -> List[Dict]:
    """
    Search Semantic Scholar API for articles
    Note: Semantic Scholar API is free and doesn't require an API key for basic usage
    """
    try:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": "title,authors,year,abstract,venue,citationCount,url"
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            papers = []
            for paper in data.get('data', []):
                papers.append({
                    "title": paper.get('title', 'N/A'),
                    "authors": [author.get('name', '') for author in paper.get('authors', [])],
                    "year": paper.get('year'),
                    "abstract": paper.get('abstract', 'No abstract available'),
                    "venue": paper.get('venue', 'N/A'),
                    "citations": paper.get('citationCount', 0),
                    "url": paper.get('url', ''),
                    "keywords": []  # Semantic Scholar doesn't provide keywords directly
                })
            return papers
        else:
            st.warning(f"Semantic Scholar API returned status {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Error fetching from Semantic Scholar: {str(e)}")
        return []

def list_available_models(api_key: str):
    """List available Gemini models that support generateContent"""
    try:
        genai.configure(api_key=api_key)
        models = genai.list_models()
        available = []
        for model in models:
            if hasattr(model, 'supported_generation_methods') and 'generateContent' in model.supported_generation_methods:
                # Extract model name (remove 'models/' prefix)
                model_name = model.name.replace('models/', '')
                available.append(model_name)
        return available
    except Exception as e:
        return []

def initialize_gemini(api_key: str):
    """Initialize Gemini AI with API key - automatically detects available models"""
    try:
        genai.configure(api_key=api_key)
        
        # Try to list available models first (may fail due to network issues)
        available_models = []
        try:
            available_models = list_available_models(api_key)
        except Exception:
            # If listing fails, continue with fallback approach
            pass
        
        if available_models:
            # Prefer models in this order
            preferred_models = [
                'gemini-1.5-flash',  # Most commonly available
                'gemini-1.5-pro',
                'gemini-2.0-flash',
                'gemini-pro'
            ]
            
            # Find first available preferred model
            for preferred in preferred_models:
                if preferred in available_models:
                    try:
                        return genai.GenerativeModel(preferred)
                    except Exception as e:
                        continue
            
            # If no preferred model found, use first available
            if available_models:
                try:
                    return genai.GenerativeModel(available_models[0])
                except Exception:
                    pass
        
        # Fallback: try common model names directly
        # These are the most common model names (try in order)
        model_names = [
            'gemini-1.5-flash',  # Most commonly available
            'gemini-1.5-pro',
            'gemini-2.0-flash',
            'gemini-pro',
            'gemini-2.5-flash'  # Newer model
        ]
        
        last_error = None
        for model_name in model_names:
            try:
                # Just create the model - don't test it here
                model = genai.GenerativeModel(model_name)
                # Store the model name for debugging
                model._model_name = model_name
                return model
            except Exception as e:
                last_error = str(e)
                continue
        
        # If all fail, log the error but return None
        if last_error:
            st.session_state.gemini_error = f"Could not initialize any model. Last error: {last_error[:200]}"
        return None
        
    except Exception as e:
        st.session_state.gemini_error = f"Error configuring Gemini: {str(e)[:200]}"
        return None

def summarize_article(article: Dict, model) -> str:
    """Use Gemini to generate a concise summary of an article"""
    if not model:
        return "Gemini API not configured. Please check your API key in .env file."
    
    try:
        prompt = f"""Please provide a concise summary (2-3 sentences) of this research paper:

Title: {article.get('title', 'N/A')}
Abstract: {article.get('abstract', 'No abstract available')}

Summary:"""
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        error_msg = str(e)
        # Provide helpful error message
        if "404" in error_msg:
            model_name = getattr(model, '_model_name', 'unknown')
            return f"❌ Model '{model_name}' not found. Error: {error_msg[:200]}"
        elif "403" in error_msg:
            return f"❌ API key permission error. Please check your API key. Error: {error_msg[:200]}"
        else:
            return f"❌ Error generating summary: {error_msg[:200]}"

def score_relevance(article: Dict, research_question: str, model) -> Dict:
    """Use Gemini to score article relevance to a research question"""
    if not model:
        return {"score": 0, "reasoning": "Gemini API not configured"}
    
    try:
        prompt = f"""Rate the relevance of this research paper to the following research question on a scale of 1-10, and provide a brief explanation:

Research Question: {research_question}

Paper Title: {article.get('title', 'N/A')}
Abstract: {article.get('abstract', 'No abstract available')}

Please respond in this format:
Score: [1-10]
Reasoning: [brief explanation]"""
        
        response = model.generate_content(prompt)
        text = response.text
        
        # Parse response
        score = 0
        reasoning = text
        if "Score:" in text:
            try:
                score = int(text.split("Score:")[1].split()[0])
            except:
                pass
        if "Reasoning:" in text:
            reasoning = text.split("Reasoning:")[1].strip()
        
        return {"score": score, "reasoning": reasoning}
    except Exception as e:
        return {"score": 0, "reasoning": f"Error: {str(e)}"}

def ai_find_relevant_papers(user_query: str, articles: List[Dict], model, top_n: int = 5) -> Dict:
    """Use Gemini AI to analyze user query and find/recommend relevant papers"""
    if not model:
        return {"error": "Gemini API not configured"}
    
    try:
        # Create a summary of available articles
        articles_summary = "\n\n".join([
            f"Paper {i+1}:\nTitle: {article.get('title', 'N/A')}\nAbstract: {article.get('abstract', 'No abstract available')[:200]}..."
            for i, article in enumerate(articles[:20])  # Limit to first 20 for context
        ])
        
        prompt = f"""You are a research assistant helping to find relevant academic papers. 

User's research interest/query: {user_query}

Available papers:
{articles_summary}

Please:
1. Analyze the user's query and identify key research themes
2. Recommend the top {top_n} most relevant papers from the list above (by their number)
3. For each recommended paper, explain why it's relevant
4. Suggest additional search terms or keywords that might help find more papers

Format your response as:
ANALYSIS: [Your analysis of the research interest]
RECOMMENDATIONS:
Paper [number]: [Title]
Relevance: [Why this paper is relevant]
[Repeat for top papers]

SEARCH_SUGGESTIONS: [Suggested keywords or search terms]"""
        
        response = model.generate_content(prompt)
        return {"response": response.text, "success": True}
    except Exception as e:
        return {"error": f"Error: {str(e)}", "success": False}

def ai_generate_search_query(user_query: str, model) -> Dict:
    """Use Gemini to generate optimized search terms from natural language query"""
    if not model:
        return {"error": "Gemini API not configured. Please check your API key in .env file.", "success": False}
    
    try:
        prompt = f"""Convert this research interest into optimized search terms for academic paper databases:

User query: {user_query}

Please provide:
1. Main search query (concise, 3-5 key terms)
2. Alternative search terms/variations
3. Related keywords

Format:
MAIN_QUERY: [optimized search query]
ALTERNATIVES: [alternative search terms]
KEYWORDS: [related keywords]"""
        
        response = model.generate_content(prompt)
        return {"response": response.text, "success": True}
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg:
            model_name = getattr(model, '_model_name', 'unknown')
            return {"error": f"Model '{model_name}' not found. Please check your API key and model availability. Full error: {error_msg[:300]}", "success": False}
        return {"error": f"Error: {error_msg[:300]}", "success": False}

def filter_articles(articles: List[Dict], search_query: str) -> List[Dict]:
    """Filter articles based on search query"""
    if not search_query:
        return articles
    
    query_lower = search_query.lower()
    filtered = []
    for article in articles:
        # Search in title, abstract, authors, and keywords
        title_match = query_lower in article.get('title', '').lower()
        abstract_match = query_lower in article.get('abstract', '').lower()
        authors_match = any(query_lower in author.lower() for author in article.get('authors', []))
        keywords_match = any(query_lower in keyword.lower() for keyword in article.get('keywords', []))
        
        if title_match or abstract_match or authors_match or keywords_match:
            filtered.append(article)
    
    return filtered

def display_article(article: Dict, index: int, gemini_model=None, research_question: str = None):
    """Display a single article card"""
    with st.container():
        col1, col2 = st.columns([0.95, 0.05])
        
        with col1:
            st.markdown(f"### {article.get('title', 'Untitled')}")
            
            # Authors and metadata
            authors_str = ", ".join(article.get('authors', []))
            metadata = f"**Authors:** {authors_str} | **Year:** {article.get('year', 'N/A')} | **Venue:** {article.get('venue', 'N/A')} | **Citations:** {article.get('citations', 0)}"
            st.markdown(metadata)
            
            # AI-powered relevance score (if Gemini is configured and research question provided)
            if gemini_model and research_question:
                if f"relevance_{index}" not in st.session_state:
                    with st.spinner("Analyzing relevance..."):
                        relevance = score_relevance(article, research_question, gemini_model)
                        st.session_state[f"relevance_{index}"] = relevance
                else:
                    relevance = st.session_state[f"relevance_{index}"]
                
                score = relevance.get('score', 0)
                color = "🟢" if score >= 7 else "🟡" if score >= 4 else "🔴"
                st.markdown(f"{color} **Relevance Score:** {score}/10")
                with st.expander("See reasoning"):
                    st.write(relevance.get('reasoning', 'No reasoning available'))
            
            # Abstract
            abstract = article.get('abstract', 'No abstract available')
            st.markdown(f"**Abstract:** {abstract[:300]}{'...' if len(abstract) > 300 else ''}")
            
            # AI summary button (if Gemini is configured)
            if gemini_model:
                if st.button(f"🤖 AI Summary", key=f"summary_btn_{index}"):
                    with st.spinner("Generating AI summary..."):
                        summary = summarize_article(article, gemini_model)
                        st.info(summary)
            
            # Keywords
            keywords = article.get('keywords', [])
            if keywords:
                keywords_str = " | ".join([f"`{kw}`" for kw in keywords])
                st.markdown(f"**Keywords:** {keywords_str}")
            
            # URL
            url = article.get('url', '')
            if url:
                st.markdown(f"[View Paper]({url})")
        
        with col2:
            is_selected = index in st.session_state.selected_articles
            if st.checkbox("Select", key=f"select_{index}", value=is_selected):
                if index not in st.session_state.selected_articles:
                    st.session_state.selected_articles.append(index)
            else:
                if index in st.session_state.selected_articles:
                    st.session_state.selected_articles.remove(index)
        
        st.divider()

# Main app
col1, col2 = st.columns([3, 1])
with col1:
    st.title("📚 Literature Review Assistant")
    st.markdown("Search and select relevant articles for your literature review")
with col2:
    st.metric("Selected Articles", len(st.session_state.selected_articles))

# Initialize Gemini model automatically from .env
env_api_key = os.getenv("GEMINI_API_KEY", "")
gemini_model = None
if env_api_key:
    gemini_model = initialize_gemini(env_api_key)
    # Show error if initialization failed
    if not gemini_model:
        error_msg = st.session_state.get('gemini_error', 'Unknown error')
        with st.expander("⚠️ Gemini API Error (Click to see details)", expanded=False):
            st.error(error_msg)
            st.info("💡 **Troubleshooting:**")
            st.markdown("""
            1. Verify your API key is correct in `.env` file
            2. Check your internet connection
            3. Make sure the API key has proper permissions
            4. Try using a different model name manually
            """)

# Main content area
tab1, tab2 = st.tabs(["🔍 Search & Browse", "📋 Selected Articles"])

with tab1:
    # Show model debug info if Gemini is configured
    if env_api_key and not gemini_model:
        available = list_available_models(env_api_key)
        if available:
            st.info(f"💡 **Tip:** Available Gemini models detected: {', '.join(available[:3])}. The app will try to use one automatically.")
    
    # Data source selection
    data_source = st.radio(
        "Data Source",
        ["Local JSON File", "Semantic Scholar API"],
        help="Choose between local JSON file or live search via Semantic Scholar API",
        horizontal=True
    )
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        min_year = st.number_input("Minimum Year", value=2020, min_value=1900, max_value=2025)
    with col2:
        min_citations = st.number_input("Minimum Citations", value=0, min_value=0)
    
    st.divider()
    
    # AI Assistant for finding relevant papers
    if gemini_model:
        st.subheader("🤖 AI Research Assistant")
        ai_query = st.text_area(
            "Describe what papers you're looking for",
            placeholder="E.g., 'I'm researching transformer architectures for natural language processing, specifically focusing on attention mechanisms and efficiency improvements'",
            help="Describe your research interest in natural language. The AI will help find relevant papers and suggest search terms.",
            height=100
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 Generate Search Terms", use_container_width=True):
                if ai_query:
                    with st.spinner("🤖 AI is analyzing your query..."):
                        result = ai_generate_search_query(ai_query, gemini_model)
                        if result.get("success"):
                            st.success("**AI-Generated Search Suggestions:**")
                            st.markdown(result["response"])
                            # Extract main query if possible
                            if "MAIN_QUERY:" in result["response"]:
                                suggested_query = result["response"].split("MAIN_QUERY:")[1].split("\n")[0].strip()
                                if st.button(f"Use: {suggested_query[:50]}..."):
                                    st.session_state.suggested_search = suggested_query
                        else:
                            st.error(result.get("error", "Failed to generate search terms"))
                else:
                    st.warning("Please enter your research interest first")
        
        with col2:
            if st.button("📚 Find Relevant Papers", use_container_width=True):
                if ai_query:
                    if st.session_state.articles_data:
                        with st.spinner("🤖 AI is analyzing papers..."):
                            result = ai_find_relevant_papers(ai_query, st.session_state.articles_data, gemini_model)
                            if result.get("success"):
                                st.success("**AI Recommendations:**")
                                st.markdown(result["response"])
                                st.session_state.ai_recommendations = result["response"]
                            else:
                                st.error(result.get("error", "Failed to analyze papers"))
                    else:
                        st.warning("Please load articles first (select data source and search)")
                else:
                    st.warning("Please enter your research interest first")
        
        # Research question for AI relevance scoring
        research_question = st.text_input(
            "🎯 Research Question (for relevance scoring on articles)",
            placeholder="Enter a specific research question to get relevance scores on each article...",
            help="Optional: Enter a research question to get AI-powered relevance scores displayed on each article"
        )
    else:
        st.info("💡 Add `GEMINI_API_KEY` to your .env file to enable AI features")
        research_question = None
    
    st.divider()
    
    # Search bar
    search_query = st.text_input(
        "🔍 Search Articles",
        placeholder="Search by title, author, keywords, or abstract...",
        help="Enter keywords to search through articles",
        value=st.session_state.get("suggested_search", "")
    )
    
    # Clear suggested search after using it
    if "suggested_search" in st.session_state:
        del st.session_state.suggested_search
    
    # Display AI recommendations if available
    if "ai_recommendations" in st.session_state and st.session_state.ai_recommendations:
        with st.expander("🤖 AI Recommendations (Click to view)", expanded=True):
            st.markdown(st.session_state.ai_recommendations)
            if st.button("Clear Recommendations", key="clear_ai_recs"):
                del st.session_state.ai_recommendations
                st.rerun()
    
    # Load articles based on data source
    if data_source == "Local JSON File":
        if not st.session_state.articles_data or st.button("🔄 Reload Articles"):
            st.session_state.articles_data = load_json_articles("articles.json")
            st.success(f"Loaded {len(st.session_state.articles_data)} articles from JSON file")
    else:  # Semantic Scholar API
        if search_query:
            with st.spinner("Searching Semantic Scholar..."):
                st.session_state.articles_data = search_semantic_scholar(search_query, limit=20)
                if st.session_state.articles_data:
                    st.success(f"Found {len(st.session_state.articles_data)} articles")
        elif not st.session_state.articles_data:
            st.info("Enter a search query to fetch articles from Semantic Scholar API")
    
    # Filter articles
    if st.session_state.articles_data:
        # Create list of (index, article) tuples to preserve original indices
        articles_with_indices = [(i, article) for i, article in enumerate(st.session_state.articles_data)]
        
        # Filter by search query
        if search_query:
            filtered_with_indices = [
                (idx, article) for idx, article in articles_with_indices
                if article in filter_articles([article], search_query)
            ]
        else:
            filtered_with_indices = articles_with_indices
        
        # Apply additional filters
        filtered_with_indices = [
            (idx, article) for idx, article in filtered_with_indices
            if (article.get('year', 0) or 0) >= min_year and (article.get('citations', 0) or 0) >= min_citations
        ]
        
        st.subheader(f"Found {len(filtered_with_indices)} article(s)")
        
        # Display articles
        for original_idx, article in filtered_with_indices:
            display_article(article, original_idx, gemini_model, research_question)
    else:
        st.info("No articles loaded. Select a data source and search for articles.")

with tab2:
    st.header("Selected Articles")
    
    if st.session_state.selected_articles:
        selected_data = [st.session_state.articles_data[i] for i in st.session_state.selected_articles if i < len(st.session_state.articles_data)]
        
        if st.button("🗑️ Clear Selection"):
            st.session_state.selected_articles = []
            st.rerun()
        
        # Display selected articles
        for idx, article_idx in enumerate(st.session_state.selected_articles):
            if article_idx < len(st.session_state.articles_data):
                article = st.session_state.articles_data[article_idx]
                with st.expander(f"{idx + 1}. {article.get('title', 'Untitled')}", expanded=True):
                    st.markdown(f"**Authors:** {', '.join(article.get('authors', []))}")
                    st.markdown(f"**Year:** {article.get('year', 'N/A')} | **Venue:** {article.get('venue', 'N/A')} | **Citations:** {article.get('citations', 0)}")
                    st.markdown(f"**Abstract:** {article.get('abstract', 'No abstract available')}")
                    if article.get('url'):
                        st.markdown(f"[View Paper]({article.get('url')})")
        
        # Export options
        st.divider()
        st.subheader("📥 Export Selected Articles")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📄 Export to JSON"):
                export_data = {
                    "export_date": datetime.now().isoformat(),
                    "selected_count": len(selected_data),
                    "articles": selected_data
                }
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(export_data, indent=2),
                    file_name="selected_articles.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📊 Export to CSV"):
                df = pd.DataFrame(selected_data)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="selected_articles.csv",
                    mime="text/csv"
                )
    else:
        st.info("No articles selected yet. Go to the Search & Browse tab to select articles.")
