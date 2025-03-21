import requests
from bs4 import BeautifulSoup
from readability import Document    # type: ignore
import re
from urllib.parse import urljoin
import openai
import streamlit as st
import pandas as pd
import time
import io

# Set page configuration
st.set_page_config(
    page_title="Wood Couture - AI Market Scout",
    page_icon="🪵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Keys - In production, these should be stored securely using st.secrets
SERPER_API_KEY = st.secrets.get("SERPER_API_KEY", None)
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)

# Google_search: Search SERPER API with a query and offset.
def google_search(query, offset=0):
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY}
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

# Get_website_content: Fetch HTML content using a custom User-Agent, timeout, and retries.
def get_website_content(website_url, timeout=20, retries=3):
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
                st.warning(f"Attempt {attempt}: Failed to fetch {website_url} with status code {response.status_code}")
        except requests.exceptions.RequestException as e:
            st.warning(f"Attempt {attempt}: Error fetching {website_url}: {e}")
    return ""

# Extract_main_content: Extract main content from HTML using readability (fallback to BeautifulSoup).
def extract_main_content(html):
    try:
        doc = Document(html)
        summary_html = doc.summary()
        soup = BeautifulSoup(summary_html, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        st.warning(f"Error using readability: {e}")
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator=" ", strip=True)

# extract_contact_details: Extract email addresses from HTML using mailto links and regex.
def extract_contact_details(html):
    emails = set()
    phones = set()
    soup = BeautifulSoup(html, "html.parser")
    
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
    
    # Extract phone numbers
    phone_regex = r'(\+?\d[\d\s\-().]{7,}\d)'
    phone_matches = re.findall(phone_regex, html)
    for phone in phone_matches:
        # Basic filtering to avoid dates and random numbers
        if len(re.sub(r'[\s\-().]+', '', phone)) >= 7:
            phones.add(phone.strip())
        
    return list(emails), list(phones)

# Find_relevant_links: Find anchor tags in the homepage whose text contains any given keywords.
def find_relevant_links(home_html, base_url, keywords):
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

# Scrape_manufacturer_website: Scrape homepage and related pages (About, Products, Contact, etc.) and combine content.
def scrape_manufacturer_website(website_url):
    with st.spinner(f"Scraping website content from {website_url}..."):
        homepage_html = get_website_content(website_url)
        if not homepage_html:
            return "", [], []
        
        homepage_content = extract_main_content(homepage_html)
        # Also extract emails from the homepage HTML
        homepage_emails, homepage_phones = extract_contact_details(homepage_html)
        
        keywords = ['about', 'products', 'contact', 'contact us', 'services', 'portfolio', 'get in touch']
        relevant_links = find_relevant_links(homepage_html, website_url, keywords)
        extracted_content = {"Homepage": homepage_content}
        
        all_emails = set(homepage_emails)
        all_phones = set(homepage_phones)
        
        # Create a section for contact details if found on the homepage
        if homepage_emails:
            extracted_content["Contact Details"] = "Emails: " + ", ".join(homepage_emails)
        
        for key, link in relevant_links.items():
            page_html = get_website_content(link)
            if page_html:
                page_content = extract_main_content(page_html)
                extracted_content[key.capitalize()] = page_content
                
                # Extract contact details from all pages
                page_emails, page_phones = extract_contact_details(page_html)
                all_emails.update(page_emails)
                all_phones.update(page_phones)
                        
        combined_content = ""
        for section, content in extracted_content.items():
            combined_content += f"\n--- {section} ---\n{content}\n"
        return combined_content, list(all_emails), list(all_phones)

# Generate_manufacturer_summary_from_content: Pass extracted content to the LLM to generate a detailed summary.
def generate_manufacturer_summary_from_content(company_name, extracted_content):
    with st.spinner(f"Generating detailed summary for {company_name}..."):
        prompt = f"""
        You are a fine-tuned business research assistant. Based on the following extracted website content from '{company_name}', generate a detailed summary that includes:
        
        - Company Size
        - Years in Business
        - Types of Products
        - Client Portfolio
        - Industry Certifications
        - Product Catalogues
        - Manufacturing Capabilities
        - Quality Standards
        - Export Information
        - Contact Details (ensure to include email addresses, phone numbers, and physical addresses if available)

        Use the information provided only in the text below and do not add any invented details.
        
        Extracted Content:
        {extracted_content}

        Please output the final summary in a clear, professional, and structured manner.
        """
        
        if not OPENAI_API_KEY:
            return "API key not available for summary generation. Please configure your OPENAI_API_KEY."
            
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=750,
                temperature=0.7,
            )
            summary = response['choices'][0]['message']['content'].strip()
            return summary
        except Exception as e:
            st.warning(f"Error generating summary: {e}")
            return "Could not generate summary due to API error. Please check your API key or try again later."

# Is_aggregator_title: Check if a title indicates an aggregator page.
def is_aggregator_title(title):
    blacklist_keywords = ['top', 'best', 'guide', 'list', 'review']
    return any(kw.lower() in title.lower() for kw in blacklist_keywords)

# Extract_manufacturer_info: Extract basic details (LinkedIn and official website) filtering out aggregator pages.
def extract_manufacturer_info(company_name):
    with st.spinner(f"Finding web presence for {company_name}..."):
        linkedin_url = None
        linkedin_query = f"{company_name} LinkedIn"
        linkedin_results = google_search(linkedin_query)
        if 'organic' in linkedin_results:
            for result in linkedin_results['organic']:
                if "linkedin.com" in result.get('link', ''):
                    linkedin_url = result['link']
                    break
        website_url = None
        website_query = f"{company_name} official website"
        website_results = google_search(website_query)
        if 'organic' in website_results:
            for result in website_results['organic']:
                title = result.get('title', '')
                if is_aggregator_title(title):
                    continue
                website_url = result.get('link')
                if website_url:
                    break
        return linkedin_url, website_url

# Extract_linkedin_details: Scrape LinkedIn page to extract phone number and location.
def extract_linkedin_details(linkedin_url):
    with st.spinner(f"Extracting details from LinkedIn..."):
        html = get_website_content(linkedin_url)
        if not html:
            return None, None
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        phone_match = re.search(r'(\+?\d[\d\s\-]{7,}\d)', text)
        phone = phone_match.group(1) if phone_match else None
        location_match = re.search(r'Location\s*[:\-]?\s*([A-Za-z0-9,\s\-]+)', text)
        location = location_match.group(1) if location_match else None
        return phone, location

# Export results to Excel
def export_results_to_excel(results):
    with st.spinner("Preparing Excel export..."):
        output = io.BytesIO()
        
        # Create a DataFrame with basic info first
        basic_data = []
        for r in results:
            basic_data.append({
                'Company': r.get('Company', ''),
                'Website': r.get('Website', ''),
                'LinkedIn': r.get('LinkedIn', ''),
                'Email': r.get('Email', ''),
                'Phone': r.get('Phone', ''),
                'Location': r.get('Location', '')
            })
            
        # Create Excel writer with openpyxl
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Write basic info to first sheet
            basic_df = pd.DataFrame(basic_data)
            basic_df.to_excel(writer, sheet_name='Company Summary', index=False)
            
            # Write detailed info to second sheet
            detailed_data = []
            for r in results:
                detailed_data.append({
                    'Company': r.get('Company', ''),
                    'Website': r.get('Website', ''),
                    'LinkedIn': r.get('LinkedIn', ''),
                    'Email': r.get('Email', ''),
                    'Phone': r.get('Phone', ''),
                    'Location': r.get('Location', ''),
                    'All Emails': ', '.join(r.get('All_Emails', [])),
                    'All Phones': ', '.join(r.get('All_Phones', [])),
                    'Summary': r.get('Summary', '')
                })
            
            detailed_df = pd.DataFrame(detailed_data)
            detailed_df.to_excel(writer, sheet_name='Detailed Information', index=False)
        
        output.seek(0)
        return output

# General search for multiple manufacturers
def general_search(country, search_terms, requirements, max_results, offset=0, existing_companies=None):
    if existing_companies is None:
        existing_companies = {}
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    manufacturers = {}
    
    for i, term in enumerate(search_terms):
        status_text.text(f"Searching for: {term}")
        query = f"{term} {requirements} in {country}".strip()
        search_results = google_search(query, offset=offset)
        
        if not search_results or 'organic' not in search_results:
            progress_bar.progress((i + 1) / len(search_terms) / 2)
            continue
            
        for j, result in enumerate(search_results.get('organic', [])):
            company_title = result.get('title', '')
            if is_aggregator_title(company_title):
                continue
                
            company_name = company_title
            website_url = result.get('link')
            
            if website_url and any(excluded in website_url for excluded in [
                "alibaba.com", "thomasnet.com", "yellowpages", "quora.com", 
                "made-in-china.com", "reddit.com", "facebook.com", "globalsources.com",
                "homedepot.com"
            ]):
                continue
                
            if company_name and company_name not in manufacturers and company_name not in existing_companies:
                status_text.text(f"Analyzing: {company_name}")
                linkedin_url, official_website = extract_manufacturer_info(company_name)
                
                if not official_website:
                    continue
                    
                manufacturers[company_name] = {
                    "Company": company_name,
                    "Website": official_website,
                    "LinkedIn": linkedin_url
                }
                
            progress_percent = ((i + (j / len(search_results.get('organic', [])))) / len(search_terms) / 2) + 0.5
            progress_bar.progress(min(progress_percent, 1.0))
            
            if len(manufacturers) >= max_results:
                break
                
        if len(manufacturers) >= max_results:
            break
    
    progress_bar.progress(1.0)
    status_text.empty()
    
    if not manufacturers:
        st.error("No manufacturers found. Please try different search parameters.")
        return []
    
    # Process results
    results = []
    for i, (company_name, details) in enumerate(manufacturers.items()):
        status_text.text(f"Processing {i+1}/{len(manufacturers)}: {company_name}")
        progress_bar.progress((i) / len(manufacturers))
        
        website_url = details.get("Website")
        linkedin_url = details.get("LinkedIn")
        
        if not website_url:
            continue
            
        extracted_content, all_emails, all_phones = scrape_manufacturer_website(website_url)
        
        if not extracted_content:
            continue
            
        summary = generate_manufacturer_summary_from_content(company_name, extracted_content)
        phone, location = None, None
        
        if linkedin_url:
            phone, location = extract_linkedin_details(linkedin_url)
            
        primary_email = all_emails[0] if all_emails else None
        primary_phone = all_phones[0] if all_phones else phone
        
        results.append({
            "Company": company_name,
            "LinkedIn": linkedin_url,
            "Website": website_url,
            "Phone": primary_phone,
            "Email": primary_email,
            "Location": location,
            "All_Emails": all_emails,
            "All_Phones": all_phones,
            "Summary": summary
        })
        
        progress_bar.progress((i + 1) / len(manufacturers))
    
    progress_bar.empty()
    status_text.empty()
    
    return results

# Specific company search
def specific_company_search(company_name):
    if not company_name:
        st.error("Please enter a company name.")
        return None
        
    with st.spinner(f"Searching for {company_name}..."):
        linkedin_url, website_url = extract_manufacturer_info(company_name)
        
        if not website_url:
            st.error("Could not find a valid website for the specified company.")
            return None
            
        extracted_content, all_emails, all_phones = scrape_manufacturer_website(website_url)
        
        if not extracted_content:
            st.error("Failed to extract content from the website.")
            return None
            
        phone, location = None, None
        
        if linkedin_url:
            phone, location = extract_linkedin_details(linkedin_url)
            
        summary = generate_manufacturer_summary_from_content(company_name, extracted_content)
        
        primary_email = all_emails[0] if all_emails else None
        primary_phone = all_phones[0] if all_phones else phone
        
        return {
            "Company": company_name,
            "LinkedIn": linkedin_url,
            "Website": website_url,
            "Phone": primary_phone,
            "Email": primary_email,
            "Location": location,
            "All_Emails": all_emails,
            "All_Phones": all_phones,
            "Summary": summary
        }

# Streamlit UI
def main():
    # Title and description
    st.title("🪵 Wood Couture AI Market Scout")
    st.subheader("Find and analyze wood manufacturer companies worldwide")
    
    # Sidebar with app options and API status
    st.sidebar.title("API Status")
    
    if not SERPER_API_KEY:
        st.sidebar.error("⚠️ SERPER_API_KEY not found. Please check your configuration.")
    else:
        st.sidebar.success("✅ SERPER_API_KEY configured")
        
    if not OPENAI_API_KEY:
        st.sidebar.warning("⚠️ OPENAI_API_KEY not found. Advanced summary generation will be disabled.")
    else:
        st.sidebar.success("✅ OPENAI_API_KEY configured")
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    ### About
    Wood Couture Market Scout helps you find and analyze wood manufacturer 
    companies worldwide. The tool uses advanced web scraping
    and AI to extract detailed information about companies.
    """)
    
    # Main content tabs
    tab1, tab2 = st.tabs(["General Search", "Company Search"])
    
    with tab1:
        st.header("Search for Multiple Companies")
        
        col1, col2 = st.columns(2)
        with col1:
            country = st.text_input("Country", "Italy")
        with col2:
            requirements = st.text_input("Specific Requirements (optional)", "")
            
        col3, col4 = st.columns(2)
        with col3:
            max_results = st.number_input("Maximum Results", min_value=1, max_value=50, value=5)
        with col4:
            offset = st.number_input("Search Offset", min_value=0, max_value=100, value=0, step=10)
        
        default_search_terms = [
            "Luxury wood furniture manufacturer",
            "Premium wood manufacturing",
            "Custom wood furniture manufacturer"
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
        
        if st.button("🔍 Search for Companies", type="primary"):
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
            existing_companies = {r['Company']: r for r in st.session_state.general_search_results}
            
            results = general_search(
                country, search_terms, 
                requirements=requirements, 
                offset=offset, 
                max_results=max_results,
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
            
            # Create a DataFrame for basic info
            basic_data = []
            for r in st.session_state.general_search_results:
                basic_data.append({
                    'Company': r.get('Company', ''),
                    'Website': r.get('Website', ''),
                    'LinkedIn': r.get('LinkedIn', ''),
                    'Email': r.get('Email', ''),
                    'Phone': r.get('Phone', '')
                })
            
            basic_df = pd.DataFrame(basic_data)
            st.dataframe(basic_df, use_container_width=True)
            
            # Display detailed results
            for i, result in enumerate(st.session_state.general_search_results):
                with st.expander(f"📋 {result['Company']} - Detailed Information"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### Contact Information")
                        if result['Website']:
                            st.markdown(f"🌐 **Website**: [{result['Website']}]({result['Website']})")
                        
                        if result['LinkedIn']:
                            st.markdown(f"👔 **LinkedIn**: [{result['LinkedIn']}]({result['LinkedIn']})")
                            
                        if result['Phone']:
                            st.markdown(f"📞 **Phone**: {result['Phone']}")
                            
                        if result['Email']:
                            st.markdown(f"📧 **Email**: {result['Email']}")
                            
                        if result['Location']:
                            st.markdown(f"📍 **Location**: {result['Location']}")
                        
                        # Display all emails if more than one
                        if 'All_Emails' in result and result['All_Emails'] and len(result['All_Emails']) > 1:
                            st.markdown("#### All Email Addresses")
                            for email in result['All_Emails']:
                                st.markdown(f"- {email}")
                                
                        # Display all phone numbers if more than one
                        if 'All_Phones' in result and result['All_Phones'] and len(result['All_Phones']) > 1:
                            st.markdown("#### All Phone Numbers")
                            for phone in result['All_Phones']:
                                st.markdown(f"- {phone}")
                    
                    with col2:
                        st.markdown("### Company Summary")
                        st.markdown(result['Summary'])
            
            # Add "Load More" button if we have search parameters
            if st.session_state.search_params:
                if st.button("🔄 Load More Results"):
                    # Calculate new offset
                    new_offset = offset + st.session_state.total_results_loaded
                    
                    # Convert existing results to dict for duplicate checking
                    existing_companies = {r['Company']: r for r in st.session_state.general_search_results}
                    
                    # Search for more results
                    more_results = general_search(
                        st.session_state.search_params['country'],
                        st.session_state.search_params['search_terms'],
                        requirements=st.session_state.search_params['requirements'],
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
        
        if st.button("🔍 Search Company", type="primary"):
            if company_name:
                result = specific_company_search(company_name)
                
                if result:
                    st.session_state.specific_company_result = result
                    st.success(f"Found information for {result['Company']}")
            else:
                st.error("Please enter a company name.")
                
        # Display company result if exists
        if 'specific_company_result' in st.session_state:
            result = st.session_state.specific_company_result
            st.markdown("---")
            
            # Add export button for single company result
            col1, col2 = st.columns([1, 4])
            with col1:
                excel_data = export_results_to_excel([st.session_state.specific_company_result])
                st.download_button(
                    label="📊 Export to Excel",
                    data=excel_data,
                    file_name=f"{result['Company']}_details.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            with col2:
                st.subheader(f"Company Profile: {result['Company']}")
            
            # Display company card
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Contact Information")
                if result['Website']:
                    st.markdown(f"🌐 **Website**: [{result['Website']}]({result['Website']})")
                
                if result['LinkedIn']:
                    st.markdown(f"👔 **LinkedIn**: [{result['LinkedIn']}]({result['LinkedIn']})")
                    
                if result['Phone']:
                    st.markdown(f"📞 **Phone**: {result['Phone']}")
                    
                if result['Email']:
                    st.markdown(f"📧 **Email**: {result['Email']}")
                    
                if result['Location']:
                    st.markdown(f"📍 **Location**: {result['Location']}")
                
                # Display all emails if more than one
                if 'All_Emails' in result and result['All_Emails'] and len(result['All_Emails']) > 1:
                    st.markdown("#### All Email Addresses")
                    for email in result['All_Emails']:
                        st.markdown(f"- {email}")
                        
                # Display all phone numbers if more than one
                if 'All_Phones' in result and result['All_Phones'] and len(result['All_Phones']) > 1:
                    st.markdown("#### All Phone Numbers")
                    for phone in result['All_Phones']:
                        st.markdown(f"- {phone}")
            
            with col2:
                st.markdown("### Company Summary")
                st.markdown(result['Summary'])
    
    # Footer
    st.markdown("---")
    st.markdown("Created by Wood Couture | AI Market Scout")

if __name__ == "__main__":
    main()
