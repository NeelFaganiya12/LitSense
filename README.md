# Literature Review Assistant 📚

A Streamlit application for searching, organizing, and managing academic papers with AI-powered clustering, ranking, and relevance analysis.

## Features

### 🔍 **Dual Data Sources**
- **Search Online**: Live search using OpenAlex API (no API key required, generous rate limits)
- **Local Papers**: Load and filter papers from your local `papers.json` file

### 🤖 **AI-Powered Features** (Gemini API)
- **Intelligent Clustering**: Automatically groups papers into thematic clusters (Local Papers only)
- **Relevance Ranking**: Ranks papers by relevance to your research topic
- **Relevance Explanation**: Get AI-generated explanations of why a paper is relevant to your research

### 📊 **Multi-Column Interface**
- **Left Column**: Search & filter controls, data source selection, year filters
- **Middle Column**: Results displayed in tabs:
  - **Clusters Tab**: Papers organized into thematic groups (Local Papers only)
  - **Review Queue Tab**: Papers ranked by AI-determined relevance
- **Right Column**: Detailed paper view with metadata, abstract, and actions

### 📌 **Paper Management**
- **Reading List**: Save papers to a persistent reading list
- **Feedback System**: Mark papers as relevant or not relevant with optional notes
- **Feedback Summary**: View all papers you've marked as relevant/not relevant in organized tabs

### 💾 **Smart Caching**
- Search results are cached to reduce API calls
- Recent searches are accessible for quick access
- Seamless switching between data sources

## Installation

1. **Install required packages:**
```bash
pip install -r requirements.txt
```

2. **Set up your Gemini API keys** (for AI features):
   - Create `.env` file in the project root
   - Add up to 3 API keys for load balancing:
     ```
     GEMINI_API_KEY_1=your_first_api_key_here
     GEMINI_API_KEY_2=your_second_api_key_here
     GEMINI_API_KEY_3=your_third_api_key_here
     ```
   - Get your API keys from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - The app will automatically rotate between available keys

3. **Prepare local papers** (optional):
   - Create `papers.json` file with your papers in the following format:
     ```json
     {
       "references": [
         {
           "id": 1,
           "title": "Paper Title",
           "authors": ["Author 1", "Author 2"],
           "year": 2023,
           "journal": "Journal Name",
           "abstract": "Paper abstract...",
           "keywords": ["keyword1", "keyword2"],
           "doi": "10.1234/example"
         }
       ]
     }
     ```

## Usage

### Running the Application

```bash
streamlit run app.py
```

### Search Online Mode

1. Select **"Search Online"** as the data source
2. Enter your research topic (e.g., "machine learning", "transformer architectures")
3. Click **"🔎 Search"** button
4. Wait 10-15 seconds for AI processing (ranking papers by relevance)
5. Browse papers in the **Review Queue** tab (ranked by relevance)
6. Click **"View"** on any paper to see detailed information
7. Use **"➕ Add to List"** to save papers to your reading list
8. Click **"🤖 Explain Relevance"** to get AI-generated relevance explanations

### Local Papers Mode

1. Select **"Local Papers"** as the data source
2. All papers from `papers.json` will be displayed automatically
3. Optionally use the filter box to search within local papers
4. Browse papers in two tabs:
   - **Clusters Tab**: Papers organized into thematic groups (collapsed by default)
   - **Review Queue Tab**: Papers ranked by relevance
5. Click on paper titles or **"View"** button to see details
6. Use the same features as Search Online mode

### Paper Details Panel

When you click **"View"** on a paper, the right column shows:
- **Title and Authors**: Full paper information
- **Publication Details**: Journal, year, DOI, citation count
- **Abstract**: Full abstract text
- **Keywords/Fields of Study**: Research areas
- **Actions**:
  - **➕ Add to List**: Save to reading list
  - **🤖 Explain Relevance**: Get AI explanation
- **Feedback Section**:
  - Mark as **✅ Relevant** or **❌ Not Relevant**
  - Add optional notes
  - View your feedback summary at the bottom of the page

### Reading List

- Located at the bottom of the page
- Shows all papers you've added using "Add to List"
- Remove papers with the **"Remove"** button
- Persists across searches and sessions

### Feedback Summary

- View papers you've marked as relevant or not relevant
- Organized in tabs: **✅ Relevant** and **❌ Not Relevant**
- Shows your notes for each paper
- Click **"View"** to jump to paper details

## File Structure

```
streamlit/
├── app.py              # Main application
├── papers.json         # Local papers data (your file)
├── articles.json       # Sample articles (legacy, not used)
├── requirements.txt    # Python dependencies
├── .env                # API keys (create this file)
├── .env.example        # Template for .env file
└── README.md           # This file
```

## Data Sources

### Search Online (OpenAlex API)
- **Free**: No API key required
- **Rate Limits**: 10 requests per second (very generous)
- **Coverage**: Millions of academic papers across all disciplines
- **Fields**: Title, authors, year, abstract, journal, citations, DOI, concepts, URL
- **Documentation**: https://docs.openalex.org/

### Local Papers (`papers.json`)
- Load papers from your local JSON file
- Instant access (no API calls)
- Supports filtering by title, abstract, or keywords
- Same AI features as online search (clustering, ranking, explanations)

## AI Features

### Clustering (Local Papers Only)
- Automatically groups papers into 3-5 thematic clusters
- Each cluster shows:
  - Cluster name
  - Number of papers
  - Key topics/keywords
  - Papers organized within the cluster

### Relevance Ranking
- Uses Gemini AI to rank papers by relevance to your search query
- Most relevant papers appear first
- Works for both online and local papers

### Relevance Explanation
- Click **"🤖 Explain Relevance"** on any paper
- Get a 3-4 sentence explanation of why the paper is relevant
- Uses your research topic as context

## API Keys

### Gemini API (Required for AI Features)
- **Storage**: Store in `.env` file (recommended)
- **Multiple Keys**: Supports up to 3 keys for load balancing
- **Get API Key**: https://makersuite.google.com/app/apikey
- **Used for**: 
  - Paper clustering
  - Relevance ranking
  - Relevance explanations
- **Optional**: App works without it, but AI features won't be available
- **Security**: The `.env` file is gitignored to keep your API keys secure

### OpenAlex API
- **No API Key Required**: Completely free and open
- **Rate Limits**: 10 requests/second (very generous, no blocking)
- **No Rate Limit Issues**: Unlike Semantic Scholar, OpenAlex has no strict rate limits

## Example Workflow

1. **Choose Data Source**: Select "Search Online" or "Local Papers"
2. **Search**: Enter your research topic
3. **Browse Results**: 
   - View clusters (Local Papers) or ranked queue (both modes)
   - Click "View" to see paper details
4. **Get AI Insights**:
   - See papers ranked by relevance
   - Click "Explain Relevance" for AI explanations
5. **Organize**:
   - Add papers to reading list
   - Mark papers as relevant/not relevant
   - Add notes to papers
6. **Review**:
   - Check your reading list at the bottom
   - View feedback summary (relevant/not relevant papers)

## Troubleshooting

### Search Online Issues
- **No results**: Try a different search query or check your internet connection
- **Error messages**: Check the error message for specific issues (invalid query, network errors)
- **Slow loading**: AI processing takes 10-15 seconds - this is normal

### Local Papers Issues
- **File not found**: Ensure `papers.json` exists in the project directory
- **Invalid JSON**: Check that your JSON file is properly formatted
- **No papers shown**: Verify your JSON has a "references" array with paper objects

### AI Features Issues
- **Clustering not working**: Ensure Gemini API keys are set in `.env`
- **API key errors**: 
  - Verify keys are correct in `.env`
  - Check API key quota/limits
  - Try using multiple keys for load balancing
- **"Explain Relevance" not working**: Check that you have at least one valid Gemini API key

### View Button Issues
- **Paper details not showing**: 
  - Try refreshing the page
  - Ensure you're viewing papers from the current data source
  - Switch data sources and try again

## Tips & Best Practices

1. **Use Multiple API Keys**: Set up 3 Gemini API keys in `.env` for better reliability
2. **Cache Results**: Recent searches are cached - use them to avoid re-searching
3. **Filter Local Papers**: Use the filter box to quickly find papers in your local collection
4. **Feedback Organization**: Mark papers as relevant/not relevant to build your feedback summary
5. **Reading List**: Add papers you want to review later to your reading list
6. **AI Explanations**: Use "Explain Relevance" to understand why papers are relevant to your research

## Technical Details

- **Framework**: Streamlit
- **AI Model**: Google Gemini 2.5 Flash
- **Search API**: OpenAlex (free, no key required)
- **State Management**: Streamlit session state for persistence
- **Caching**: Built-in caching for search results and paper data

## Future Enhancements

Potential features to add:
- [ ] Export reading list to BibTeX/CSV
- [ ] PDF download integration
- [ ] Citation network visualization
- [ ] Save/load review sessions
- [ ] Collaborative reviews
- [ ] Advanced filtering options
- [ ] Bibliography generation
