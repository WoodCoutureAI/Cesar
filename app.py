import streamlit as st
import requests
import re
import os
import pandas as pd
import io
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Set API keys from Streamlit secrets
# This will use secrets.toml in local development and deployed environment variables in production
SERPER_API_KEY = st.secrets.get("SERPER_API_KEY", None)

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

def extract_contact_info_from_website(website_url):
    """
    Extract contact information directly from a company's website.
    
    Args:
        website_url (str): URL of the company website
        
    Returns:
        tuple: (email, phone, location) extracted from the website
    """
    if not website_url:
        return None, None, None
        
    try:
        response = requests.get(website_url, timeout=10)
        if response.status_code != 200:
            return None, None, None

        # Parse the main homepage
        main_html = response.text
        soup = BeautifulSoup(main_html, "html.parser")
        
        # Look for a potential contact page link using keywords
        contact_page_url = None
        contact_keywords = ["contact", "contact us", "get in touch", "reach us", "support"]
        for a in soup.find_all("a", href=True):
            link_text = a.get_text(strip=True).lower()
            href = a["href"].lower()
            if any(keyword in link_text for keyword in contact_keywords) or any(keyword in href for keyword in contact_keywords):
                contact_page_url = a["href"]
                # If it's a relative URL, join it with the main URL
                if contact_page_url.startswith("/"):
                    contact_page_url = urljoin(website_url, contact_page_url)
                break

        # If a contact page is found, fetch and parse it
        if contact_page_url:
            contact_response = requests.get(contact_page_url, timeout=10)
            if contact_response.status_code == 200:
                soup = BeautifulSoup(contact_response.text, "html.parser")

        # Get the full text from the (contact) page
        full_text = soup.get_text(separator=" ", strip=True)
        
        # --- EMAIL EXTRACTION ---
        email = None
        mailto_link = soup.find("a", href=lambda href: href and href.lower().startswith("mailto:"))
        if mailto_link:
            email = mailto_link.get("href").replace("mailto:", "").strip()
        if not email:
            email_matches = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", full_text)
            email = email_matches[0] if email_matches else None

        # PHONE NUMBER EXTRACTION 
        phone = None
        tel_link = soup.find("a", href=lambda href: href and href.lower().startswith("tel:"))
        if tel_link:
            phone = tel_link.get("href").replace("tel:", "").strip()
        if not phone:
            phone_matches = re.findall(r'\+?\d[\d\s\-\(\)]{7,}\d', full_text)
            phone = phone_matches[0] if phone_matches else None

        # LOCATION EXTRACTION 
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

        return email, phone, location

    except Exception as e:
        st.error(f"Error fetching website details from {website_url}: {e}")
        return None, None, None

def extract_company_summary_from_search(company_name):
    """
    Extract a summary about the company from search results.
    
    Args:
        company_name (str): Company name to search for
        
    Returns:
        str: Summary text about the company
    """
    # First try to get company overview
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
            for result in results['organic'][:3]:  #
                snippet = result.get('snippet', '')
                if snippet and len(snippet) > 50:  # 
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

def search_specific_company(company_name):
    """
    Search for information about a specific company.
    
    Args:
        company_name (str): Name of the company to search
        
    Returns:
        dict: Company information including contact details and summary
    """
    with st.spinner(f"Searching for information about {company_name}..."):
        # Find company website
        website_query = f"{company_name} official website"
        website_results = google_search(website_query)
        website_url = None
        
        if 'organic' in website_results:
            for result in website_results['organic']:
                potential_url = result.get('link')
                if potential_url and is_valid_company_result(potential_url):
                    website_url = potential_url
                    break
        
        # Find LinkedIn profile
        linkedin_url = find_linkedin_url(company_name)
        
        # Extract contact information from website
        email, phone_number, location = extract_contact_info_from_website(website_url)
        
        # Get company summary from search results
        summary = extract_company_summary_from_search(company_name)
        
        return {
            "company_name": company_name,
            "linkedin_url": linkedin_url,
            "website_url": website_url,
            "phone_number": phone_number,
            "email": email,
            "location": location,
            "summary": summary
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
            
            # Strip common suffixes from company names for cleaner results
            if company_name:
                company_name = re.sub(r' - .*$', '', company_name)
                company_name = re.sub(r' \|.*$', '', company_name)

            # Skip if not a valid company result
            if not is_valid_company_result(website_url) or not company_name:
                continue

            if company_name not in companies:
                status_text.write(f"Processing: {company_name}")
                
                # Extract contact info from website
                email, phone_number, location = extract_contact_info_from_website(website_url)
                
                # Find LinkedIn URL
                linkedin_url = find_linkedin_url(company_name)
                
                # Get summary from search results
                summary = extract_company_summary_from_search(company_name)
                
                companies[company_name] = {
                    "company_name": company_name,
                    "website_url": website_url,
                    "linkedin_url": linkedin_url,
                    "phone_number": phone_number,
                    "email": email,
                    "location": location,
                    "summary": summary
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
    
    # Make sure we only include columns that exist
    columns_to_use = [col for col in columns_order if col in df.columns]
    df = df[columns_to_use]
    
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
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    ### About
    Wood Couture Market Scout helps you find and analyze furniture 
    suppliers or manufacturers worldwide. The tool uses Google search 
    to extract key information about companies.
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
            
            with col2:
                st.markdown("### Company Summary")
                st.markdown(result['summary'])

if __name__ == "__main__":
    main() 