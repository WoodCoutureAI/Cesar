import streamlit as st
import requests
import re
import os
import pandas as pd
import io
from bs4 import BeautifulSoup
from urllib.parse import urljoin
try:
    from readability import Document  # For extracting main content from webpages
except ImportError:
    st.error("Please install python-readability: pip install readability-lxml")
import openai

# Set API keys from Streamlit secrets
# This will use secrets.toml in local development and deployed environment variables in production
SERPER_API_KEY = st.secrets.get("SERPER_API_KEY", None)
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)

# Configure OpenAI if API key is available
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# Set page config
st.set_page_config(
    page_title="Wood Couture Market Scout",
    page_icon="🪵",
    layout="wide"
)

# Ignored domains that are typically marketplaces or irrelevant for manufacturer searches
IGNORED_DOMAINS = [
    "alibaba.com", "thomasnet.com", "yellowpages", "quora.com", 
    "made-in-china.com", "reddit.com", "facebook.com", "flooring", 
    "globalsources.com", "lumber", "homedepot.com", "amazon.com",
    "indiamart.com", "wikipedia.org", "etsy.com", "pinterest.com"
]

def google_search(query, offset=0):
    """
    Perform a Google search using the Serper API.
    
    Args:
        query (str): Search query to execute
        offset (int): Starting position for search results
        
    Returns:
        dict: JSON response from Serper API
    """
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY}
    # Request up to 10 results starting at the offset
    params = {"q": query, "hl": "en", "start": offset, "num": 10}
    try:
        response = requests.get(url, params=params, headers=headers)
        if response.status_code != 200:
            st.error(f"SERPER API error for query '{query}': {response.text}")
            return {}
        return response.json()
    except Exception as e:
        st.error(f"Exception during SERPER API call for query '{query}': {e}")
        return {}

def is_valid_company_result(website_url):
    """
    Check if the search result is a valid company website (not a marketplace, etc.)
    
    Args:
        website_url (str): URL to check
        
    Returns:
        bool: True if valid company site, False otherwise
    """
    if not website_url:
        return False
        
    return not any(excluded in website_url.lower() for excluded in IGNORED_DOMAINS)

def is_aggregator_title(title):
    """
    Check if a title indicates an aggregator page.
    
    Args:
        title (str): Page title to check
        
    Returns:
        bool: True if title suggests an aggregator/list page
    """
    blacklist_keywords = ['top', 'best', 'guide', 'list', 'review', 'directory']
    return any(kw.lower() in title.lower() for kw in blacklist_keywords)

def extract_manufacturer_info(company_name):
    """
    Extract basic details (LinkedIn and official website) filtering out aggregator pages.
    
    Args:
        company_name (str): Name of the company to search for
        
    Returns:
        tuple: (linkedin_url, website_url) extracted from search results
    """
    # Find LinkedIn URL
    linkedin_url = None
    linkedin_query = f"{company_name} LinkedIn"
    linkedin_results = google_search(linkedin_query)
    if 'organic' in linkedin_results:
        for result in linkedin_results['organic']:
            if "linkedin.com/company/" in result.get('link', ''):
                linkedin_url = result['link']
                break
    
    # Find company website
    website_url = None
    website_query = f"{company_name} official website"
    website_results = google_search(website_query)
    if 'organic' in website_results:
        for result in website_results['organic']:
            title = result.get('title', '')
            # Skip aggregator pages
            if is_aggregator_title(title):
                continue
            potential_url = result.get('link')
            if potential_url and is_valid_company_result(potential_url):
                website_url = potential_url
                break
    
    return linkedin_url, website_url

def get_website_content(website_url, timeout=20, retries=3):
    """
    Fetch HTML content using a custom User-Agent, timeout, and retries.
    
    Args:
        website_url (str): URL to fetch content from
        timeout (int): Request timeout in seconds
        retries (int): Number of retry attempts
        
    Returns:
        str: HTML content or empty string if failed
    """
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/114.0.0.0 Safari/537.36")
    }
    session = requests.Session()
    session.headers.update(headers)
    for attempt in range(1, retries + 1):
        try:
            response = session.get(website_url, timeout=timeout)
            if response.status_code == 200:
                return response.text
            else:
                if attempt == retries:
                    st.warning(f"Failed to fetch {website_url} with status code {response.status_code}")
        except requests.exceptions.RequestException as e:
            if attempt == retries:
                st.warning(f"Error fetching {website_url}: {e}")
    return ""

def extract_main_content(html):
    """
    Extract main content from HTML using readability (fallback to BeautifulSoup).
    
    Args:
        html (str): HTML content to extract from
        
    Returns:
        str: Main content text
    """
    try:
        # Use readability if available
        doc = Document(html)
        summary_html = doc.summary()
        soup = BeautifulSoup(summary_html, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        # Fall back to standard BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator=" ", strip=True)

def extract_contact_details(html):
    """
    Extract contact details from HTML using multiple methods.
    
    Args:
        html (str): HTML content to extract from
        
    Returns:
        dict: Dictionary with email, phone and address information
    """
    soup = BeautifulSoup(html, "html.parser")
    full_text = soup.get_text(separator=" ", strip=True)
    
    # Email extraction
    emails = set()
    # Look for mailto links
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("mailto:"):
            email = href.split("mailto:")[1].split("?")[0].strip()
            if email:
                emails.add(email)
    
    # Fallback: Use regex to find email patterns in the entire HTML
    email_regex = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    found_emails = re.findall(email_regex, html)
    for email in found_emails:
        emails.add(email)
    
    # Phone extraction
    phones = set()
    # Look for tel links
    tel_links = soup.find_all("a", href=lambda href: href and href.lower().startswith("tel:"))
    for tel_link in tel_links:
        phone = tel_link.get("href").replace("tel:", "").strip()
        if phone:
            phones.add(phone)
    
    # Use regex to find phone patterns
    phone_matches = re.findall(r'\+?\d[\d\s\-\(\)]{7,}\d', full_text)
    for phone in phone_matches:
        phones.add(phone)
    
    # Location extraction
    location = None
    address_tag = soup.find("address")
    if address_tag:
        location = address_tag.get_text(separator=" ", strip=True)
    if not location:
        # Attempt to extract text following the word "Address"
        loc_match = re.search(r"Address[:\s]*([A-Za-z0-9,\s\-]+?)(?=\s*(Call us|Email|$))", full_text, re.IGNORECASE)
        if loc_match:
            location = loc_match.group(1).strip()
        else:
            loc_match = re.search(r"Address[:\s]*([A-Za-z0-9,\s\-]+)", full_text, re.IGNORECASE)
            if loc_match:
                location = loc_match.group(1).strip()
    
    return {
        "emails": list(emails),
        "phones": list(phones),
        "location": location
    }

def find_relevant_links(home_html, base_url, keywords):
    """
    Find anchor tags in the homepage whose text contains any given keywords.
    
    Args:
        home_html (str): HTML content of the homepage
        base_url (str): Base URL for resolving relative links
        keywords (list): List of keywords to search for in link text
        
    Returns:
        dict: Dictionary mapping keywords to their URLs
    """
    soup = BeautifulSoup(home_html, "html.parser")
    links = {}
    for a in soup.find_all("a", href=True):
        link_text = a.get_text().strip().lower()
        href = a['href']
        for kw in keywords:
            if kw in link_text and kw not in links:
                full_url = urljoin(base_url, href)
                links[kw] = full_url
    return links

def scrape_manufacturer_website(website_url):
    """
    Scrape homepage and related pages (About, Products, Contact, etc.) and combine content.
    
    Args:
        website_url (str): URL of the manufacturer's website
        
    Returns:
        dict: Dictionary with extracted content from various pages and contact details
    """
    if not website_url:
        return {"content": "", "contact_details": {}}
    
    homepage_html = get_website_content(website_url)
    if not homepage_html:
        return {"content": "", "contact_details": {}}
    
    homepage_content = extract_main_content(homepage_html)
    homepage_contact = extract_contact_details(homepage_html)
    
    keywords = ['about', 'products', 'contact', 'contact us', 'services', 'portfolio', 'get in touch']
    relevant_links = find_relevant_links(homepage_html, website_url, keywords)
    
    extracted_content = {"Homepage": homepage_content}
    contact_details = homepage_contact
    
    for key, link in relevant_links.items():
        page_html = get_website_content(link)
        if page_html:
            page_content = extract_main_content(page_html)
            extracted_content[key.capitalize()] = page_content
            
            # If this page is a contact page, also extract contact details
            if "contact" in key.lower():
                page_contact = extract_contact_details(page_html)
                # Merge contact details
                contact_details["emails"] = list(set(contact_details.get("emails", []) + page_contact.get("emails", [])))
                contact_details["phones"] = list(set(contact_details.get("phones", []) + page_contact.get("phones", [])))
                if not contact_details.get("location") and page_contact.get("location"):
                    contact_details["location"] = page_contact["location"]
    
    # Format as dictionary
    combined_content = ""
    for section, content in extracted_content.items():
        combined_content += f"\n--- {section} ---\n{content}\n"
    
    return {
        "content": combined_content,
        "contact_details": contact_details
    }

def generate_manufacturer_summary(company_name, extracted_content):
    """
    Generate a detailed manufacturer summary using OpenAI.
    
    Args:
        company_name (str): Company name
        extracted_content (str): Extracted website content
        
    Returns:
        str: Detailed company summary
    """
    # If OpenAI API key is not available, return a simplified summary
    if not OPENAI_API_KEY:
        # Return a simplified version
        summary = f"Information about {company_name}"
        if len(extracted_content) > 500:
            summary += ": " + extracted_content[:500] + "..."
        else:
            summary += ": " + extracted_content
        return summary
    
    # Format the prompt
    prompt = f"""
    You are a fine-tuned business research assistant. Based on the following extracted website content from '{company_name}', generate a detailed summary that includes:
    
    - Company Size
    - Years in Business
    - Types of Products
    - Client Portfolio
    - Industry Certifications
    - Manufacturing Capabilities
    - Quality Standards
    - Export Information

    Use the information provided only in the text below and do not add any invented details.
    If some information is not available, simply omit that section.
    
    Extracted Content:
    {extracted_content}

    Please output the final summary in a clear, professional, and structured manner.
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # Using a more affordable model
            messages=[{"role": "user", "content": prompt}],
            max_tokens=750,
            temperature=0.7,
        )
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        st.error(f"Error generating summary with OpenAI: {e}")
        # Fallback to a basic summary
        summary = f"Information about {company_name}"
        if len(extracted_content) > 500:
            summary += ": " + extracted_content[:500] + "..."
        else:
            summary += ": " + extracted_content
        return summary

def extract_company_summary_from_search(company_name):
    """
    Extract a summary about the company from search results.
    Used as a fallback if scraping and OpenAI summary generation fails.
    
    Args:
        company_name (str): Company name to search for
        
    Returns:
        str: Summary text about the company
    """
    query = f"{company_name} company about overview"
    search_results = google_search(query)
    
    summary_texts = []
    
    # Try multiple search queries to get comprehensive information
    queries = [
        f"{company_name} company about overview",
        f"{company_name} about us company description",
        f"{company_name} company profile business"
    ]
    
    for query in queries:
        results = google_search(query)
        if 'organic' in results:
            # Extract snippets from search results
            for result in results['organic'][:3]:
                snippet = result.get('snippet', '')
                if snippet and len(snippet) > 50:
                    # Clean up the snippet
                    cleaned_snippet = re.sub(r'\s+', ' ', snippet).strip()
                    if cleaned_snippet not in summary_texts:  # Avoid duplicates
                        summary_texts.append(cleaned_snippet)
            
            # If we have knowledge graph info, use that
            if 'knowledgeGraph' in results:
                kg = results['knowledgeGraph']
                if 'description' in kg:
                    kg_desc = kg['description']
                    if kg_desc not in summary_texts:
                        summary_texts.insert(0, kg_desc)  # Prioritize knowledge graph
    
    if summary_texts:
        # Combine snippets into a coherent summary
        summary = " ".join(summary_texts)
        # Clean up the summary
        summary = re.sub(r'\s+', ' ', summary).strip()
        return summary
    
    return "No information available."

def find_linkedin_url(company_name):
    """
    Find LinkedIn URL for a company.
    
    Args:
        company_name (str): Company name to search for
        
    Returns:
        str: LinkedIn URL if found, None otherwise
    """
    linkedin_query = f"{company_name} LinkedIn company page"
    linkedin_results = google_search(linkedin_query)
    
    if 'organic' in linkedin_results:
        for result in linkedin_results['organic']:
            link = result.get('link', '')
            if "linkedin.com/company/" in link:
                return link
    
    return None

def extract_linkedin_details(linkedin_url):
    """
    Extract details from a company's LinkedIn profile.
    
    Args:
        linkedin_url (str): LinkedIn URL to scrape
        
    Returns:
        dict: Dictionary with extracted LinkedIn details
    """
    if not linkedin_url:
        return {"phone": None, "location": None}
        
    try:
        html = get_website_content(linkedin_url)
        if not html:
            return {"phone": None, "location": None}
            
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        
        # Try to find phone number
        phone_match = re.search(r'(\+?\d[\d\s\-]{7,}\d)', text)
        phone = phone_match.group(1) if phone_match else None
        
        # Try to find location
        location_match = re.search(r'Location\s*[:\-]?\s*([A-Za-z0-9,\s\-]+)', text)
        location = location_match.group(1).strip() if location_match else None
        
        return {"phone": phone, "location": location}
    except Exception as e:
        st.warning(f"Error extracting LinkedIn details: {e}")
        return {"phone": None, "location": None}

def search_specific_company(company_name):
    """
    Search for detailed information about a specific company.
    
    Args:
        company_name (str): Name of the company to search
        
    Returns:
        dict: Company information including contact details and summary
    """
    with st.spinner(f"Searching for information about {company_name}..."):
        # Get company website and LinkedIn URLs using the new helper function
        linkedin_url, website_url = extract_manufacturer_info(company_name)
        
        # Scrape the manufacturer's website
        st.info(f"Scraping website content for {company_name}...")
        scraped_data = scrape_manufacturer_website(website_url)
        
        # Extract LinkedIn details
        linkedin_details = extract_linkedin_details(linkedin_url)
        
        # Generate company summary using OpenAI if available
        if scraped_data["content"]:
            st.info("Generating detailed company summary...")
            summary = generate_manufacturer_summary(company_name, scraped_data["content"])
        else:
            # Fallback to search-based summary
            summary = extract_company_summary_from_search(company_name)
        
        # Consolidate contact details from website and LinkedIn
        contact_details = scraped_data["contact_details"]
        email = contact_details.get("emails", [None])[0] if contact_details.get("emails") else None
        phone_number = contact_details.get("phones", [None])[0] if contact_details.get("phones") else None
        location = contact_details.get("location")
        
        # If no phone/location from website, use LinkedIn data
        if not phone_number and linkedin_details.get("phone"):
            phone_number = linkedin_details["phone"]
        if not location and linkedin_details.get("location"):
            location = linkedin_details["location"]
        
        return {
            "company_name": company_name,
            "linkedin_url": linkedin_url,
            "website_url": website_url,
            "phone_number": phone_number,
            "email": email,
            "location": location,
            "summary": summary,
            "all_emails": contact_details.get("emails", []),
            "all_phones": contact_details.get("phones", [])
        }

def search_multiple_companies(country, search_terms, additional_requirements="", offset=0, max_results=20, existing_companies=None):
    """
    Find information for multiple companies based on search criteria.
    
    Args:
        country (str): Country to search in
        search_terms (list): List of search terms
        additional_requirements (str): Additional search requirements
        offset (int): Search result offset
        max_results (int): Maximum number of results to return
        existing_companies (dict): Existing companies to avoid duplicates when loading more
        
    Returns:
        list: List of company information dictionaries
    """
    companies = existing_companies if existing_companies is not None else {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Track how many new companies we've found
    new_companies_count = 0
    
    # Perform searches for each term
    for i, term in enumerate(search_terms):
        query = f"{term} {additional_requirements} in {country}".strip()
        status_text.write(f"Searching: {query}")
        search_results = google_search(query, offset=offset)

        if not search_results or 'organic' not in search_results:
            continue  # Skip if no results found

        for j, result in enumerate(search_results['organic']):
            company_name = result.get('title')
            website_url = result.get('link')
            
            # Skip aggregator pages
            if is_aggregator_title(company_name):
                continue
            
            # Strip common suffixes from company names for cleaner results
            if company_name:
                company_name = re.sub(r' - .*$', '', company_name)
                company_name = re.sub(r' \|.*$', '', company_name)

            # Skip if not a valid company result
            if not is_valid_company_result(website_url) or not company_name:
                continue

            if company_name not in companies:
                status_text.write(f"Processing: {company_name}")
                
                # Get LinkedIn URL using the helper function instead of separate search
                linkedin_url, better_website_url = extract_manufacturer_info(company_name)
                
                # Use the better website URL if available
                if better_website_url:
                    website_url = better_website_url
                
                # Scrape the manufacturer's website
                scraped_data = scrape_manufacturer_website(website_url)
                
                # Extract LinkedIn details
                linkedin_details = extract_linkedin_details(linkedin_url)
                
                # Generate company summary
                summary = ""
                if scraped_data["content"]:
                    summary = generate_manufacturer_summary(company_name, scraped_data["content"])
                else:
                    # Fallback to search-based summary
                    summary = extract_company_summary_from_search(company_name)
                
                # Consolidate contact details
                contact_details = scraped_data["contact_details"]
                email = contact_details.get("emails", [None])[0] if contact_details.get("emails") else None
                phone_number = contact_details.get("phones", [None])[0] if contact_details.get("phones") else None
                location = contact_details.get("location")
                
                # If no phone/location from website, use LinkedIn data
                if not phone_number and linkedin_details.get("phone"):
                    phone_number = linkedin_details["phone"]
                if not location and linkedin_details.get("location"):
                    location = linkedin_details["location"]
                
                companies[company_name] = {
                    "company_name": company_name,
                    "website_url": website_url,
                    "linkedin_url": linkedin_url,
                    "phone_number": phone_number,
                    "email": email,
                    "location": location,
                    "summary": summary,
                    "all_emails": contact_details.get("emails", []),
                    "all_phones": contact_details.get("phones", [])
                }
                
                new_companies_count += 1
                
                # Update progress based on new companies found
                progress = min(1.0, new_companies_count / max_results)
                progress_bar.progress(progress)

                if new_companies_count >= max_results:
                    break
        
        if new_companies_count >= max_results:
            break

    if not companies:
        st.warning("No companies found. Please try different search parameters.")
        return []

    # Convert dict to list for return
    results = list(companies.values())
    
    progress_bar.progress(1.0)
    status_text.empty()
    return results


def export_results_to_excel(results):
    """
    Export search results to Excel file
    
    Args:
        results (list): List of company information dictionaries
        
    Returns:
        bytes: Excel file as bytes for download
    """
    # Create a DataFrame from the results
    df = pd.DataFrame(results)
    
    # Reorder columns for better readability
    columns_order = [
        "company_name", "website_url", "linkedin_url", 
        "phone_number", "email", "location", "summary"
    ]
    
    # Additional columns if they exist
    additional_columns = ["all_emails", "all_phones"]
    for col in additional_columns:
        if col in df.columns:
            columns_order.append(col)
    
    # Make sure we only include columns that exist
    columns_to_use = [col for col in columns_order if col in df.columns]
    df = df[columns_to_use]
    
    # Format list columns to improve readability in Excel
    for col in ["all_emails", "all_phones"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
    
    # Create a buffer for the Excel file
    buffer = io.BytesIO()
    
    # Use pandas to save the DataFrame to the buffer as an Excel file
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Companies')
        
        # Auto-adjust columns' width
        worksheet = writer.sheets['Companies']
        for i, col in enumerate(df.columns):
            # Get the maximum length in this column
            max_len = max(df[col].astype(str).apply(len).max(), len(col)) + 2
            worksheet.set_column(i, i, max_len)
    
    # Important: Move to the beginning of the buffer before returning
    buffer.seek(0)
    return buffer.getvalue()

# Streamlit UI
def main():
    st.title("🪵 Wood Couture Market Scout")
    st.subheader("Find and analyze furniture suppliers or manufacturers worldwide")
    
    # Sidebar for API key status
    st.sidebar.title("API Status")
    
    if not SERPER_API_KEY:
        st.sidebar.error("⚠️ SERPER_API_KEY not found. Please check your secrets.toml file.")
    else:
        st.sidebar.success("✅ SERPER_API_KEY configured")
        
    if not OPENAI_API_KEY:
        st.sidebar.warning("⚠️ OPENAI_API_KEY not found. Advanced summary generation will be disabled.")
    else:
        st.sidebar.success("✅ OPENAI_API_KEY configured")
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    ### About
    Wood Couture Market Scout helps you find and analyze furniture 
    suppliers or manufacturers worldwide. The tool uses advanced web scraping
    and AI to extract detailed information about companies.
    """)
    
    # Main content tabs
    tab1, tab2 = st.tabs(["General Search", "Company Search"])
    
    with tab1:
        st.header("Search for Multiple Companies")
        
        col1, col2 = st.columns(2)
        with col1:
            country = st.text_input("Country", "United States")
        with col2:
            requirements = st.text_input("Specific Requirements (optional)", "")
            
        col3, col4 = st.columns(2)
        with col3:
            max_results = st.number_input("Maximum Results", min_value=1, max_value=50, value=5)
        with col4:
            offset = st.number_input("Search Offset", min_value=0, max_value=100, value=0, step=10)
        
        default_search_terms = [
            "Luxury wood furniture manufacturer",
            "High-end wood supplier",
            "Premium wood manufacturing",
            "Custom wood furniture manufacturer",
            "Top wood manufacturers"
        ]
        
        custom_terms = st.text_area("Custom Search Terms (one per line, leave empty to use defaults)", "")
        search_terms = default_search_terms
        if custom_terms.strip():
            search_terms = [term for term in custom_terms.strip().split("\n") if term.strip()]
        
        # Initialize session state for storing results
        if 'general_search_results' not in st.session_state:
            st.session_state.general_search_results = []
            st.session_state.total_results_loaded = 0
            st.session_state.search_params = None
        
        if st.button("🔍 Search for Companies"):
            if not SERPER_API_KEY:
                st.error("Please configure your SERPER_API_KEY in the secrets.toml file before searching.")
                return
                
            # Show search terms being used
            st.write("Using search terms:")
            for term in search_terms:
                st.write(f"- {term}")
            
            # Reset results when starting a new search
            st.session_state.general_search_results = []
            st.session_state.total_results_loaded = 0
            
            # Store search parameters
            st.session_state.search_params = {
                'country': country,
                'search_terms': search_terms,
                'requirements': requirements
            }
            
            # Convert existing results to dict for duplicate checking
            existing_companies = {r['company_name']: r for r in st.session_state.general_search_results}
            
            results = search_multiple_companies(
                country, search_terms, additional_requirements=requirements, 
                offset=offset, max_results=max_results,
                existing_companies=existing_companies
            )
            
            if results:
                st.session_state.general_search_results = results
                st.session_state.total_results_loaded = len(results)
                st.success(f"Found {len(results)} companies matching your criteria")
        
        # Display results if they exist
        if st.session_state.general_search_results:
            st.markdown("---")
            
            # Add export button
            col1, col2 = st.columns([1, 4])
            with col1:
                excel_data = export_results_to_excel(st.session_state.general_search_results)
                st.download_button(
                    label="📊 Export to Excel",
                    data=excel_data,
                    file_name="wood_couture_search_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            with col2:
                st.subheader("Search Results")
            
            # Display results
            for i, result in enumerate(st.session_state.general_search_results):
                with st.expander(f"{i+1}. {result['company_name']}"):
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.markdown("### Contact Information")
                        if result['website_url']:
                            st.markdown(f"🌐 **Website**: [{result['website_url']}]({result['website_url']})")
                        if result['linkedin_url']:
                            st.markdown(f"👔 **LinkedIn**: [{result['linkedin_url']}]({result['linkedin_url']})")
                        if result['phone_number']:
                            st.markdown(f"📞 **Phone**: {result['phone_number']}")
                        if result['email']:
                            st.markdown(f"📧 **Email**: {result['email']}")
                        if result['location']:
                            st.markdown(f"📍 **Location**: {result['location']}")
                            
                        # Display all emails and phones if available
                        if 'all_emails' in result and result['all_emails'] and len(result['all_emails']) > 1:
                            st.markdown("#### All Email Addresses")
                            for email in result['all_emails']:
                                st.markdown(f"- {email}")
                                
                        if 'all_phones' in result and result['all_phones'] and len(result['all_phones']) > 1:
                            st.markdown("#### All Phone Numbers")
                            for phone in result['all_phones']:
                                st.markdown(f"- {phone}")
                    
                    with col2:
                        st.markdown("### Company Summary")
                        st.markdown(result['summary'])
            
            # Add "Load More" button if we have search parameters
            if st.session_state.search_params:
                if st.button("🔄 Load More Results"):
                    # Calculate new offset
                    new_offset = st.session_state.total_results_loaded
                    
                    # Convert existing results to dict for duplicate checking
                    existing_companies = {r['company_name']: r for r in st.session_state.general_search_results}
                    
                    # Search for more results
                    more_results = search_multiple_companies(
                        st.session_state.search_params['country'],
                        st.session_state.search_params['search_terms'],
                        additional_requirements=st.session_state.search_params['requirements'],
                        offset=new_offset,
                        max_results=max_results,
                        existing_companies=existing_companies
                    )
                    
                    if more_results:
                        # Update results and counter
                        st.session_state.general_search_results.extend(more_results)
                        st.session_state.total_results_loaded = len(st.session_state.general_search_results)
                        st.success(f"Found {len(more_results)} additional companies")
                        st.experimental_rerun()
                    else:
                        st.info("No more results found.")
    
    with tab2:
        st.header("Search for a Specific Company")
        company_name = st.text_input("Company Name", "")
        
        if st.button("🔍 Search for Company"):
            if not company_name:
                st.warning("Please enter a company name")
                return
                
            if not SERPER_API_KEY:
                st.error("Please configure your SERPER_API_KEY in the secrets.toml file before searching.")
                return
                
            result = search_specific_company(company_name)
            st.session_state.specific_company_result = result
            st.success(f"Found information for {company_name}")
            
        # Display company result if exists
        if "specific_company_result" in st.session_state:
            result = st.session_state.specific_company_result
            st.markdown("---")
            
            # Add export button for single company result
            col1, col2 = st.columns([1, 4])
            with col1:
                excel_data = export_results_to_excel([st.session_state.specific_company_result])
                st.download_button(
                    label="📊 Export to Excel",
                    data=excel_data,
                    file_name=f"{result['company_name']}_details.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            with col2:
                st.subheader(f"Company Profile: {result['company_name']}")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("### Contact Information")
                if result['website_url']:
                    st.markdown(f"🌐 **Website**: [{result['website_url']}]({result['website_url']})")
                if result['linkedin_url']:
                    st.markdown(f"👔 **LinkedIn**: [{result['linkedin_url']}]({result['linkedin_url']})")
                if result['phone_number']:
                    st.markdown(f"📞 **Phone**: {result['phone_number']}")
                if result['email']:
                    st.markdown(f"📧 **Email**: {result['email']}")
                if result['location']:
                    st.markdown(f"📍 **Location**: {result['location']}")
                    
                # Display all emails and phones if available
                if 'all_emails' in result and result['all_emails'] and len(result['all_emails']) > 1:
                    st.markdown("#### All Email Addresses")
                    for email in result['all_emails']:
                        st.markdown(f"- {email}")
                        
                if 'all_phones' in result and result['all_phones'] and len(result['all_phones']) > 1:
                    st.markdown("#### All Phone Numbers")
                    for phone in result['all_phones']:
                        st.markdown(f"- {phone}")
            
            with col2:
                st.markdown("### Company Summary")
                st.markdown(result['summary'])

if __name__ == "__main__":
    main() 
