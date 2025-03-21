# Wood Couture Market Scout

## Overview
Wood Couture Market Scout is a powerful tool for finding and analyzing furniture suppliers or manufacturers worldwide. It leverages Google search capabilities through the Serper API to extract comprehensive information about companies.

## Features
- **General Search**: Find companies based on country and specific requirements
- **Company-Specific Search**: Get detailed information about a specific company
- **Automated Information Extraction**: Collects website URLs, LinkedIn profiles, phone numbers, email addresses, and location information
- **Direct Web Scraping**: Extracts contact information directly from company websites
- **Robust Filtering**: Ignores marketplace websites and other non-manufacturer domains
- **Excel Export**: Export search results to Excel spreadsheet

## Requirements
- Python 3.7+
- Serper API key (for Google search)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/wood-couture-market-scout.git
cd wood-couture-market-scout
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `secrets.toml` file in the `.streamlit` directory in the project root:
```bash
mkdir -p .streamlit
touch .streamlit/secrets.toml
```

4. Add your Serper API key to the `secrets.toml` file:
```toml
SERPER_API_KEY = "your_serper_api_key_here"
```

## Usage

### Streamlit Web Interface
Run the Streamlit web application:
```bash
streamlit run app.py
```

### Using the Application

1. **API Key Status**: The sidebar shows if your API key is properly configured
2. **General Search**:
   - Enter a country name
   - Add specific requirements (optional)
   - Set the maximum number of results to display
   - Use default search terms or add your custom terms
   - Click "Search for Companies"
   - Click "Export to Excel" to download the results as an Excel file

3. **Company Search**:
   - Enter a specific company name
   - Click "Search for Company"
   - View the detailed company profile
   - Click "Export to Excel" to download the company details as an Excel file

## How It Works

1. The application searches for companies using the Serper API (Google search)
2. It filters out marketplace websites and irrelevant domains
3. For each company found, it extracts contact details through web scraping
4. It collects company summaries from search results and knowledge graphs
5. Results are displayed in an organized, user-friendly interface
6. Results can be exported to Excel for further analysis

## Note
This tool requires a valid Serper API key to function. The search functionality consumes API credits from the Serper service, so use it judiciously.

## Deployment
When deploying to Streamlit Cloud, add your API key to the Streamlit Cloud secrets management feature through the dashboard.

## License
[MIT License](LICENSE) 