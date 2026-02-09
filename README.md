# Literature Review Assistant 📚

A Streamlit application for searching, selecting, and managing relevant literature review articles with AI-powered features.

## Features

- 🔍 **Search & Browse**: Search through articles by title, author, keywords, or abstract
- 📋 **Article Selection**: Select and manage relevant articles for your literature review
- 🤖 **AI-Powered Features** (with Gemini API):
  - Relevance scoring based on your research question
  - AI-generated article summaries
- 📊 **Statistics**: View analytics about your article collection
- 📥 **Export**: Export selected articles to JSON or CSV
- 🌐 **Multiple Data Sources**:
  - Local JSON file with sample articles
  - Semantic Scholar API for real-time article search

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Set up your Gemini API key (optional, for AI features):
   - Create `.env` and add your Gemini API key:
     ```
     GEMINI_API_KEY=your_actual_api_key_here
     ```
   - Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

## Usage

### Basic Usage (Local JSON File)

1. Run the application:
```bash
streamlit run app.py
```

2. The app will load articles from `articles.json` by default
3. Use the search bar to filter articles
4. Select articles using the checkboxes
5. View selected articles in the "Selected Articles" tab
6. Export your selection as JSON or CSV

### Using Semantic Scholar API

1. Select "Semantic Scholar API" as the data source in the sidebar
2. Enter a search query (e.g., "machine learning", "natural language processing")
3. The app will fetch real articles from Semantic Scholar
4. No API key required for basic usage!

### AI Features (Gemini API)

To enable AI-powered features:

1. **Set up API key** (choose one method):
   - **Method 1 (Recommended)**: Add your API key to `.env` file:
     ```
     GEMINI_API_KEY=your_api_key_here
     ```
   
2. Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

3. Once configured, the app will automatically load your API key from `.env`

4. Enter your research question in the search tab

5. Articles will be automatically scored for relevance

6. Click "AI Relevance" button on any article for an AI-generated summary

## File Structure

```
streamlit/
├── app.py  # Main application
├── articles.json              # Sample articles data
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Data Sources

### Local JSON File
The `articles.json` file contains 20 sample articles with the following structure:
- title
- authors (list)
- year
- abstract
- venue
- citations
- url
- keywords (list)

### Semantic Scholar API
- Free API (no key required for basic usage)
- Provides real academic papers
- Fields: title, authors, year, abstract, venue, citations, url

## Features Breakdown

### Search & Browse Tab
- Search articles by keywords
- Filter by year and citation count
- View article details (title, authors, abstract, etc.)
- Select articles for your review

### Selected Articles Tab
- View all selected articles
- Clear selection
- Export to JSON or CSV

### Statistics Tab
- Total articles count
- Selected articles count
- Average publication year
- Total citations
- Charts for year distribution and citations
- Top venues list

## API Keys

### Gemini API
- **Storage**: Store your API key in `.env` file (recommended) or enter manually in the app
- **Get your API key**: https://makersuite.google.com/app/apikey
- **Used for**: Relevance scoring, article summarization
- **Optional**: App works without it, but AI features won't be available
- **Security**: The `.env` file is gitignored to keep your API key secure

### Semantic Scholar API
- No API key required for basic usage
- Rate limits: 100 requests per 5 minutes (free tier)

## Example Workflow

1. **Load Articles**: Choose data source (JSON or Semantic Scholar)
2. **Search**: Enter keywords or research question
3. **Filter**: Adjust year and citation filters if needed
4. **AI Scoring** (optional): Enter research question and configure Gemini API
5. **Select**: Check articles that are relevant
6. **Review**: Go to "Selected Articles" tab to review your selection
7. **Export**: Download as JSON or CSV for further analysis

## Troubleshooting

- **Articles not loading**: Check that `articles.json` exists in the same directory
- **Semantic Scholar not working**: Check your internet connection and API status
- **Gemini errors**: Verify your API key is correct and has sufficient quota
- **Export issues**: Make sure you have selected at least one article

## Future Enhancements

Potential features to add:
- [ ] Save/load review sessions
- [ ] Collaborative reviews
- [ ] Citation network visualization
- [ ] PDF download integration
- [ ] Advanced filtering (by venue, author, etc.)
- [ ] Bibliography generation (BibTeX, etc.)
