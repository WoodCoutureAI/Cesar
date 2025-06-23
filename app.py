import streamlit as st
import requests
import re
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from readability import Document
from openai import OpenAI
import time
import json

# Set your API keys here (or better, via Streamlit secrets)
# API Keys - In production, these should be stored securely using st.secrets
SERPER_API_KEY = st.secrets.get("SERPER_API_KEY", None)
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)

# Initialize the OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# --- Configuration for Gemini-style search ---
MAX_SERPER_RESULTS_PER_QUERY = 10
MAX_PAGES_TO_SCRAPE_PER_FILTER = 3
SCRAPE_TIMEOUT = 15
SCRAPE_DELAY = 1
MAX_CONTENT_PER_PAGE = 1000
MAX_SNIPPETS_PER_CATEGORY = 3
SERPER_API_URL = "https://google.serper.dev/search"

# Updated search categories with broader search terms
search_categories = {
    "Location": [
        "company location shipping port logistics shenzhen"
    ],
    "Minimum Order Quantity (MOQ)": [
        "minimum order quantity MOQ per item",
        "minimum order quantity MOQ per project container"
    ],
    "Lead Time": [
        "sampling lead time production samples",
        "production lead time manufacturing",
        "shipping readiness delivery time"
    ],
    "Certifications & Standards": [
        "ISO FSC CE UL RoHS certification",
        "Fire-rating compliance BS5852 CAL117 certification",
        "Sustainability eco-certifications environmental"
    ],
    "Past Project Experience": [
        "hospitality hotels resorts projects completed",
        "residential projects portfolio completed"
    ],
    "Customization Capability": [
        "OEM ODM manufacturing capabilities",
        "engineering shop drawing technical support",
        "sampling prototyping capability development"
    ],
    "Production Capacity": [
        "monthly yearly production capacity output",
        "factory size workforce employees workers"
    ],
    "Quality Control Process": [
        "in-house QA QC quality control",
        "third-party inspection SGS Intertek certification",
        "factory audit report quality assessment"
    ],
    "Language/Communication": [
        "English speaking sales engineering team communication"
    ],
    "Logistics Support": [
        "export experience FOB EXW DDP shipping",
        "in-house packing team packaging",
        "crating labeling capabilities shipping"
    ],
    "Company Profile or Catalog": [
        "company profile PDF catalog download",
        "website GlobalSources profile company information"
    ],
    "Project References": [
        "hotel brands commercial clients portfolio",
        "project photos gallery portfolio images"
    ],
    "Factory Tour or Video": [
        "virtual tour factory video facility",
        "Google Maps location verified facility"
    ],
    "Ability to Handle Large-Scale Projects": [
        "multi-phase multi-location large scale projects deliveries"
    ],
    "After-Sales Support": [
        "warranty guarantee support",
        "spare parts supply replacement",
        "installation guides manual documentation"
    ],
    "Client Reviews or Ratings": [
        "customer reviews ratings testimonials references"
    ]
}

# Flatten the search categories into search filters
search_filters = []
for category, subcategories in search_categories.items():
    search_filters.extend(subcategories)

# Add these constants at the top with other configurations
MAX_GPT4_TOKENS = 6000  # Conservative limit for input to leave room for response
MAX_RETRIES = 3

# Function to scrape content from a URL
def scrape_page_content(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=SCRAPE_TIMEOUT)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')

        # Remove unwanted elements
        for element in soup.find_all(['script', 'style', 'nav', 'footer', 'iframe']):
            element.decompose()

        # Try to find main content
        content = ""
        main_content = soup.find(['main', 'article', 'div[role="main"]', '#content', '.main-content'])
        if main_content:
            content = main_content.get_text(separator=' ', strip=True)
        else:
            # Fallback to body content
            content = soup.body.get_text(separator=' ', strip=True) if soup.body else ""

        # Clean up and limit the content
        content = ' '.join(content.split())
        return content[:MAX_CONTENT_PER_PAGE]  # Limit content length
    except Exception as e:
        return None

# Function to perform Serper API search using requests
def perform_serper_search(query, api_key, num_results):
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }
    payload = json.dumps({
        "q": query,
        "num": num_results,
        "gl": "us",
        "hl": "en"
    })
    try:
        response = requests.post(SERPER_API_URL, headers=headers, data=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Serper API Request Error: {e}")
        return None
    except json.JSONDecodeError as e:
        st.error(f"Serper API JSON Decode Error: {e}. Response was: {response.text}")
        return None

# Google Search: Search SERPER API with a query and offset.
def google_search(query, offset=0):
    if not SERPER_API_KEY:
        st.error("SERPER_API_KEY is not set. Cannot perform Google searches.")
        return {}
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
        st.warning("Error using readability: " + str(e))
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator=" ", strip=True)

def extract_contact_details(html):
    emails = set()
    phones = set()
    addresses_found = []

    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=' ', strip=True)

    # Emails
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("mailto:"):
            email = href.split("mailto:")[1].split("?")[0].strip()
            if email:
                emails.add(email)

    # regex in raw HTML (fixed character class)
    email_regex = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    for email in re.findall(email_regex, html):
        emails.add(email)

    # Phones (stricter regex + post-filter)
    phone_re = re.compile(
    r'''
    \+              # must start with a plus
    \d{1,3}         # country code (1–3 digits)
    (?:[\s\-\(\)]*\d){7,12} # then 7–12 more digits, possibly separated by spaces/dashes/parens
    ''',
    re.VERBOSE
)

    phones = set()
    for candidate in phone_re.findall(html):
    # Strip all non-digits/+ and double-check digit count
        digits = re.sub(r"[^\d+]", "", candidate)
        if 8 <= len(digits) <= 15: # Standard phone numbers generally 8 to 15 digits
            phones.add(candidate.strip())

    # Addresses
    address_patterns = [
        r'\d{1,5}\s(?:[A-Za-z]+\s){1,4}'
        r'(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Dr|Boulevard|Blvd|Place|Pl|Court|Ct)'
        r'\.?,?\s*[A-Za-z\s]*,?\s*[A-Z]{2,3}\s*\d{5}(?:-\d{4})?',
        r'(?:P\.O\.\s*Box|PO\s*Box)\s*\d+',
        r'[A-Za-z\s]+(?:,\s*[A-Za-z\s]+){1,3}(?:,\s*[A-Z]{2,3})?(?:,\s*\d{4,6})?'
    ]
    for pat in address_patterns:
        for match in re.findall(pat, text, re.IGNORECASE):
            addresses_found.append(match.strip())

    addresses = list(set(addresses_found))

    return list(emails), list(phones), addresses


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

# scrape_manufacturer_website: Scrape homepage and related pages (About, Products, Contact, etc.) and combine content.
def scrape_manufacturer_website(website_url):
    with st.spinner(f"Scraping website content from {website_url}..."):
        homepage_html = get_website_content(website_url)
        if not homepage_html:
            return "", [], [], [] # Return empty lists for emails, phones, addresses if homepage fails

        homepage_content = extract_main_content(homepage_html)
        homepage_emails, homepage_phones, homepage_addresses = extract_contact_details(homepage_html)

        keywords = ['about', 'products', 'contact', 'contact us', 'services', 'portfolio', 'get in touch', 'inquiry', 'catalogue', 'company', 'profile', 'overview', 'Products & Services', 'Projects']
        relevant_links = find_relevant_links(homepage_html, website_url, keywords)
        extracted_content = {"Homepage": homepage_content}

        all_emails = set(homepage_emails)
        all_phones = set(homepage_phones)
        all_addresses = set(homepage_addresses)

        # Create a section for contact details if found on the homepage
        if homepage_emails or homepage_phones or homepage_addresses:
            contact_details_section = ""
            if homepage_emails:
                contact_details_section += "Emails: " + ", ".join(homepage_emails) + "\n"
            if homepage_phones:
                contact_details_section += "Phones: " + ", ".join(homepage_phones) + "\n"
            if homepage_addresses:
                contact_details_section += "Addresses: " + ", ".join(homepage_addresses) + "\n"
            extracted_content["Contact Details"] = contact_details_section

        for key, link in relevant_links.items():
            page_html = get_website_content(link)
            if page_html:
                page_content = extract_main_content(page_html)
                extracted_content[key.capitalize()] = page_content

                # Extract contact details from all pages
                page_emails, page_phones, page_addresses = extract_contact_details(page_html)
                all_emails.update(page_emails)
                all_phones.update(page_phones)
                all_addresses.update(page_addresses)

        combined_content = ""
        for section, content in extracted_content.items():
            combined_content += f"\n--- {section} ---\n{content}\n"
        return combined_content, list(all_emails), list(all_phones), list(all_addresses)

# generate_manufacturer_summary_from_content: Use the LLM to generate a detailed summary.
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
        - Contact Details (ensure to include email addresses, phone numbers, and physical addresses which is a must)

        Use the information provided only in the text below and do not add any invented details.

        Extracted Content:
        {extracted_content}

        Please output the final summary in a clear, professional, and structured manner.
        """

        if not OPENAI_API_KEY:
            return "API key not available for summary generation. Please configure your OPENAI_API_KEY."

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=750,
                temperature=0.7,
            )
            summary = response.choices[0].message.content.strip()
            return summary
        except Exception as e:
            return f"Could not generate summary due to API error: {e}. Please check your API key or try again later."

# Is_aggregator_title: Check if a title indicates an aggregator page.
def is_aggregator_title(title):
    blacklist_keywords = ['top', 'best', 'guide', 'list', 'review']
    return any(kw.lower() in title.lower() for kw in blacklist_keywords)

# Extract_manufacturer_info: Extract basic details (LinkedIn and official website) filtering out aggregator pages.
def extract_manufacturer_info(company_name):
    linkedin_url = None
    linkedin_query = f"{company_name} LinkedIn"
    linkedin_results = google_search(linkedin_query)
    if linkedin_results and 'organic' in linkedin_results:
        for result in linkedin_results['organic']:
            if "linkedin.com" in result.get('link', ''):
                linkedin_url = result['link']
                break
    website_url = None
    website_query = f"{company_name} official website"
    website_results = google_search(website_query)
    if website_results and 'organic' in website_results:
        for result in website_results['organic']:
            title = result.get('title', '')
            if is_aggregator_title(title):
                continue
            website_url = result.get('link')
            if website_url:
                # Basic check to filter out known bad domains early
                if any(excl in website_url for excl in [
                    "alibaba.com", "thomasnet.com", "yellowpages", "quora.com",
                    "made-in-china.com", "reddit.com", "facebook.com", "globalsources.com",
                    "homedepot.com", "indiamart.com", "justdial.com", "yelp.com", "amazon.com",
                    "ebay.com", "wikipedia.org", "tripadvisor.com"
                ]):
                    continue
                break # Found a potential official website
    return linkedin_url, website_url

# Extract_linkedin_details: Scrape LinkedIn page to extract phone number and location.
def extract_linkedin_details(linkedin_url):
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


# --- Streamlit UI ---

st.title("Wood Couture AI Market Scout")
st.markdown("""
I can help you search for and analyze bespoke furniture related manufacturer companies worldwide.
You can perform either a general search by country or search for a specific company.
""")

# Initialize session state for main data structures
if 'suppliers_for_analysis' not in st.session_state:
    st.session_state['suppliers_for_analysis'] = []
if 'perform_analysis' not in st.session_state:
    st.session_state['perform_analysis'] = False
# NEW: Store the last general search results
if 'last_general_search_results' not in st.session_state:
    st.session_state['last_general_search_results'] = {}
# NEW: Store the last specific company search result
if 'last_specific_company_summary' not in st.session_state:
    st.session_state['last_specific_company_summary'] = None
# NEW: Store the last input for specific company search to pre-fill
if 'last_specific_company_name_input' not in st.session_state:
    st.session_state['last_specific_company_name_input'] = ''
# Store analysis results
if 'analysis_results' not in st.session_state:
    st.session_state['analysis_results'] = {}
# Track which suppliers have been analyzed
if 'analyzed_suppliers' not in st.session_state:
    st.session_state['analyzed_suppliers'] = set()

# Create two tabs
search_tab, analysis_tab = st.tabs(["Search", "Analysis Results"])

# Search Tab Content
with search_tab:
    # Select search mode using a radio button
    search_mode = st.radio("Select Search Mode", ("General Search", "Specific Company Search"))

    # --- START OF GENERAL SEARCH BLOCK (Requested by user) ---
    if search_mode == "General Search":
        st.header("General Manufacturer Search")
        
        supplier_category = st.selectbox(
            "Select Supplier Category",
            [
                "Bespoke Furniture Supplier",
                "Lighting Supplier",
                "Drapery Supplier",
                "Outdoor Furniture Supplier",
                "Rugs Supplier"
            ],
            index=0
        )

        category_to_terms = {
            "Bespoke Furniture Supplier": [
                "Bespoke furniture manufacturer",
                "luxury woodwork manufacturer",
                "Custom woodwork manufacturer",
                "Premium furniture manufacturing",
                "Custom furniture manufacturer"
            ],
            "Lighting Supplier": [
                "Lighting supplier",
                "Illumination supplier",
                "Illumination manufacturer",
                "Lighting manufacturer",
                "Custom lighting manufacturer"
            ],
            "Drapery Supplier": [
                "Drapery supplier",
                "Curtain supplier",
                "Curtain manufacturer",
                "Drapery manufacturer",
                "Custom drapery manufacturer"
            ],
            "Outdoor Furniture Supplier": [
                "Outdoor furniture manufacturer",
                "Exterior furniture supplier",
                "Garden furniture supplier",
                "Patio furniture manufacturer",
                "Custom outdoor furniture supplier"
            ],
            "Rugs Supplier": [
                "Rug manufacturer",
                "Rugs supplier",
                "Carpet supplier",
                "Carpet manufacturer",
                "Custom Rug supplier"
            ]
        }

        st.session_state['search_terms'] = category_to_terms[supplier_category]

        country = st.text_input("Enter the country")
        requirements = st.text_input("Enter any specific requirements (optional)")
        max_results = st.number_input("How many companies do you want to display?", min_value=1, max_value=50, value=20)

        if st.button("Perform General Search"):
            if not SERPER_API_KEY:
                st.error("Cannot perform search: SERPER_API_KEY is not set.")
            else:
                offset = 0
                # Reset results for a new search
                st.session_state['last_general_search_results'] = {}
                st.session_state['last_specific_company_summary'] = None # Clear specific search if general search initiated

                search_terms = st.session_state['search_terms']
                search_count = 0
                for term in search_terms:
                    if search_count >= max_results:
                        break
                    
                    query = f"{term} {requirements} in {country}".strip()
                    st.info(f"Performing search: {query}")
                    search_results = google_search(query, offset=offset)
                    if not search_results or 'organic' not in search_results:
                        continue
                    
                    for result in search_results['organic']:
                        if search_count >= max_results:
                            break
                            
                        company_title = result.get('title', '')
                        if is_aggregator_title(company_title):
                            continue
                        website_url_candidate = result.get('link')
                        
                        if website_url_candidate and any(excl in website_url_candidate for excl in [
                            "alibaba.com", "thomasnet.com", "yellowpages", "quora.com",
                            "made-in-china.com", "reddit.com", "facebook.com", "globalsources.com",
                            "homedepot.com", "indiamart.com", "justdial.com", "yelp.com", "amazon.com",
                            "ebay.com", "wikipedia.org", "tripadvisor.com", "pinterest.com", "instagram.com",
                            "youtube.com", "twitter.com", "m.alibaba.com"
                        ]):
                            continue
                        
                        linkedin_url, official_website = extract_manufacturer_info(company_title)
                        if not official_website:
                            continue

                        # Store results directly into session state
                        if company_title not in st.session_state['last_general_search_results']:
                            with st.spinner(f"Scraping and summarizing {company_title} website..."):
                                combined_content, all_emails, all_phones, all_addresses = scrape_manufacturer_website(official_website)
                                if not combined_content:
                                    st.warning(f"Failed to extract meaningful content from {official_website} for {company_title}")
                                    continue
                                summary = generate_manufacturer_summary_from_content(company_title, combined_content)

                            st.session_state['last_general_search_results'][company_title] = {
                                "website_url": official_website,
                                "linkedin_url": linkedin_url,
                                "summary": summary,
                                "emails": all_emails,
                                "phones": all_phones,
                                "addresses": all_addresses,
                                "extracted_content": combined_content,
                                "name": company_title
                            }
                            search_count += 1
                            if search_count >= max_results:
                                break
                
                if not st.session_state['last_general_search_results']:
                    st.error("No manufacturers found. Please try different search parameters.")
                else:
                    st.info(f"Search complete. Found {search_count} companies.") # Indicate search completion with count
        
        # ALWAYS DISPLAY RESULTS IF THEY EXIST IN SESSION STATE
        if st.session_state['last_general_search_results']:
            st.subheader("Found Manufacturers:")
            for company_name, details in st.session_state['last_general_search_results'].items():
                st.write(f"### {company_name}")
                st.write(f"**Website:** {details['website_url']}")
                if details['linkedin_url']:
                    st.write(f"**LinkedIn:** {details['linkedin_url']}")
                st.markdown(details['summary'])
                
                if st.button(f"Add {company_name} to Analysis Box", key=f"add_general_{company_name.replace(' ', '_')}"):
                    supplier_data = details.copy() # Create a copy to avoid modifying the cached search result directly
                    if not any(s['website_url'] == supplier_data['website_url'] for s in st.session_state['suppliers_for_analysis']):
                        st.session_state['suppliers_for_analysis'].append(supplier_data)
                        st.success(f"{company_name} added to analysis box!")
                    else:
                        st.info(f"{company_name} is already in the analysis box.")
                st.markdown("---")
    # --- END OF GENERAL SEARCH BLOCK ---

    elif search_mode == "Specific Company Search":
        st.header("Specific Company Search")
        # Pre-fill with last searched company name
        specific_company_input = st.text_input("Enter the name of the specific company/manufacturer",
                                               value=st.session_state['last_specific_company_name_input'])

        if st.button("Search Specific Company", key="search_specific_company_btn"):
            if not SERPER_API_KEY:
                st.error("Cannot perform search: SERPER_API_KEY is not set.")
            elif specific_company_input:
                st.session_state['last_specific_company_name_input'] = specific_company_input # Store for next rerun
                st.session_state['last_general_search_results'] = {} # Clear general search if specific search initiated
                st.session_state['last_specific_company_summary'] = None # Clear previous specific search result

                linkedin_url, website_url = extract_manufacturer_info(specific_company_input)
                if not website_url:
                    st.error("Could not find a valid website for the specified company.")
                    st.session_state['last_specific_company_summary'] = None
                else:
                    st.write(f"**Website:** {website_url}")
                    if linkedin_url:
                        st.write(f"**LinkedIn:** {linkedin_url}")
                    with st.spinner("Scraping website and generating summary..."):
                        combined_content, all_emails, all_phones, all_addresses = scrape_manufacturer_website(website_url)
                        if not combined_content:
                            st.error("Failed to extract content from the website.")
                            st.session_state['last_specific_company_summary'] = None
                        else:
                            summary = generate_manufacturer_summary_from_content(specific_company_input, combined_content)

                            # Store specific company details in session state
                            st.session_state['last_specific_company_summary'] = {
                                "name": specific_company_input,
                                "website_url": website_url,
                                "linkedin_url": linkedin_url,
                                "summary": summary,
                                "emails": all_emails,
                                "phones": all_phones,
                                "addresses": all_addresses,
                                "extracted_content": combined_content
                            }
                    st.info("Search complete. Results are displayed below.")
            else:
                st.error("No company provided. Please enter a company name.")

        # ALWAYS DISPLAY SPECIFIC COMPANY RESULTS IF THEY EXIST IN SESSION STATE
        if st.session_state['last_specific_company_summary']:
            details = st.session_state['last_specific_company_summary']
            st.subheader(f"Manufacturer: {details['name']}")
            st.write(f"**Website:** {details['website_url']}")
            if details['linkedin_url']:
                st.write(f"**LinkedIn:** {details['linkedin_url']}")
            st.markdown(details['summary'])

            if st.button(f"Add {details['name']} to Analysis Box", key=f"add_specific_{details['name'].replace(' ', '_')}"):
                supplier_data = details # All data is already in 'details'
                if not any(s['website_url'] == supplier_data['website_url'] for s in st.session_state['suppliers_for_analysis']):
                    st.session_state['suppliers_for_analysis'].append(supplier_data)
                    st.success(f"{details['name']} added to analysis box!")
                else:
                    st.info(f"{details['name']} is already in the analysis box.")
            st.markdown("---")
        elif specific_company_input and not st.session_state['last_specific_company_summary']:
            pass

# Analysis Tab Content
with analysis_tab:
    st.header("Supplier Analysis Results")
    
    if not st.session_state['suppliers_for_analysis']:
        st.info("No suppliers to analyze. Please add suppliers from the Search tab first.")
    else:
        # Show suppliers available for analysis
        st.subheader("Suppliers Available for Analysis")
        
        # Create columns for the header
        col1, col2, col3, col4 = st.columns([0.5, 0.15, 0.15, 0.2])
        with col1:
           st.markdown("**Company Name**")
        with col2:
           st.markdown("**Analyze**")
        with col3:
            st.markdown("**Status**")
        with col4:
            st.markdown("**Remove**")
        
        st.markdown("---")
        
        # List all suppliers with their analysis status
        for i, supplier in enumerate(st.session_state['suppliers_for_analysis']):
            col1, col2, col3, col4 = st.columns([0.5, 0.15, 0.15, 0.2])
            
            with col1:
                st.write(f"{supplier['name']}")
            
            with col2:
                if st.button("Analyze", key=f"analyze_btn_{i}"):
                    st.session_state['current_supplier_index'] = i
                    st.session_state['perform_analysis'] = True
                    st.rerun()
            
            with col3:
                if supplier['name'] in st.session_state['analyzed_suppliers']:
                    st.success("✓")
                else:
                    st.info("Pending")
            
            with col4:
                if st.button("Remove", key=f"remove_analysis_btn_{i}"):
                    # Remove from analyzed suppliers if present
                    if supplier['name'] in st.session_state['analyzed_suppliers']:
                        st.session_state['analyzed_suppliers'].remove(supplier['name'])
                    # Remove from analysis results if present
                    if supplier['name'] in st.session_state['analysis_results']:
                        del st.session_state['analysis_results'][supplier['name']]
                    # Remove from suppliers list
                    st.session_state['suppliers_for_analysis'].pop(i)
                    st.rerun()
            
            # If this supplier has already been analyzed, show a button to view results
            if supplier['name'] in st.session_state['analyzed_suppliers']:
                if st.button("View Analysis", key=f"view_analysis_{i}"):
                    st.session_state['view_supplier_name'] = supplier['name']
                    st.rerun()
        
        st.markdown("---")
        
        # Button to analyze all suppliers
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Analyze All Suppliers", key="analyze_all_btn"):
                st.session_state['perform_analysis'] = True
                st.session_state['analyze_all'] = True
                st.rerun()
        
        with col2:
            if st.button("Clear All Analysis Results", key="clear_all_results"):
                st.session_state['analyzed_suppliers'] = set()
                st.session_state['analysis_results'] = {}
                st.rerun()
        
        # Display analysis results if requested
        if 'view_supplier_name' in st.session_state and st.session_state['view_supplier_name'] in st.session_state['analysis_results']:
            st.markdown("---")
            supplier_name = st.session_state['view_supplier_name']
            st.subheader(f"Analysis Results for {supplier_name}")
            
            # Find the supplier data
            supplier_data = None
            for s in st.session_state['suppliers_for_analysis']:
                if s['name'] == supplier_name:
                    supplier_data = s
                    break
            
            if supplier_data:
                st.write(f"**Website:** {supplier_data['website_url']}")
                if supplier_data.get('linkedin_url'):
                    st.write(f"**LinkedIn:** {supplier_data['linkedin_url']}")
                
                # Display the analysis results
                with st.expander("View Detailed Analysis", expanded=True):
                    #st.markdown("```\n" + st.session_state['analysis_results'][supplier_name] + "\n```")
                    st.markdown(analysis_result)
                
                if st.button("Back to Supplier List"):
                    if 'view_supplier_name' in st.session_state:
                        del st.session_state['view_supplier_name']
                    st.rerun()
            else:
                st.error(f"Could not find supplier data for {supplier_name}")
        
        # Perform analysis if requested
        if 'perform_analysis' in st.session_state and st.session_state['perform_analysis']:
            st.markdown("---")
            st.subheader("Analysis in Progress")
            
            # Determine which suppliers to analyze
            suppliers_to_analyze = []
            if st.session_state.get('analyze_all', False):
                suppliers_to_analyze = st.session_state['suppliers_for_analysis']
            elif 'current_supplier_index' in st.session_state:
                index = st.session_state['current_supplier_index']
                if 0 <= index < len(st.session_state['suppliers_for_analysis']):
                    suppliers_to_analyze = [st.session_state['suppliers_for_analysis'][index]]
            
            for supplier in suppliers_to_analyze:
                st.subheader(f"Analyzing: {supplier['name']}")
                st.write(f"**Website:** {supplier['website_url']}")
                if supplier.get('linkedin_url'):
                    st.write(f"**LinkedIn:** {supplier['linkedin_url']}")

                with st.spinner(f"Performing deep analysis for {supplier['name']}..."):
                    all_collected_data = {}
                    progress_bar = st.progress(0)
                    num_filters = len(search_filters)

                    for i, filter_query in enumerate(search_filters):
                        query = f"{supplier['name']} {filter_query}"
                        relevant_snippets = []
                        relevant_links = []
                        scraped_content = ""

                        search_response = perform_serper_search(query, SERPER_API_KEY, MAX_SERPER_RESULTS_PER_QUERY)
                        
                        if search_response and 'organic' in search_response:
                            # Collect limited snippets and links
                            for result in search_response['organic'][:MAX_SNIPPETS_PER_CATEGORY]:
                                if result.get('snippet'):
                                    relevant_snippets.append(result.get('snippet', '')[:200])  # Limit snippet length
                                relevant_links.append(result.get('link', ''))

                            # Strategic Content Extraction (Scraping Top Links)
                            scraped_count = 0
                            for link in relevant_links:
                                if scraped_count >= MAX_PAGES_TO_SCRAPE_PER_FILTER:
                                    break
                                if any(link.endswith(ext) for ext in ['.pdf', '.doc', '.docx', '.jpg', '.png', '.mp4']):
                                    continue
                                
                                content = scrape_page_content(link)
                                if content:
                                    scraped_content += f"\n\n--- Content from {link} ---\n{content}"
                                    scraped_count += 1
                                    time.sleep(SCRAPE_DELAY)

                        all_collected_data[filter_query] = {
                            "snippets": relevant_snippets,
                            "links": relevant_links[:3],  # Limit stored links
                            "scraped_content": scraped_content[:2000]  # Limit stored content
                        }
                        
                        progress_bar.progress((i + 1) / num_filters)

                    # Format the collected data more concisely
                    collected_data_text = ""
                    for filter_query, data in all_collected_data.items():
                        collected_data_text += f"Category: {filter_query}\n"
                        if data["snippets"]:
                            collected_data_text += "Key Information:\n" + "\n".join(
                                f"- {snippet[:150]}..." for snippet in data["snippets"][:2]
                            ) + "\n"
                        collected_data_text += "\n"

                    # Add the already scraped content from initial search
                    if supplier.get('extracted_content'):
                        collected_data_text += "\n--- Previously Extracted Content ---\n"
                        collected_data_text += supplier['extracted_content']

                    # Create the final prompt with detailed format
                    final_summary_prompt = f"""Based on the collected information about the company '{supplier['name']}', 
provide a detailed analysis in the following specific format. For each item, provide specific details instead of just yes/no answers. Include numbers, descriptions, and specific capabilities where available:

Location / Region
Specify the exact city, district, and describe the proximity to major ports or logistics hubs in detail.

Minimum Order Quantity (MOQ):-
-Per item: Specify exact minimum quantities for different product categories
-Per project or container: Specify container or project-based minimums with details

Lead Time:-
-Sampling lead time: Specify exact timeframes in days/weeks
-Production lead time: Specify timeframes for different order sizes
-Shipping readiness: Detail preparation and shipping times

Certifications & Standards:-
-ISO, FSC, CE, UL, RoHS: List specific certification numbers and validity dates if available
-Fire-rating compliance (BS5852, CAL117): Detail specific compliance levels and test results
-Sustainability/eco-certifications: List specific certifications with details

Past Project Experience:-
-Hospitality (hotels, resorts): Name specific projects, sizes, and locations
-Residential: Detail types and scales of residential projects

Customization Capability:-
-OEM vs ODM: Specify exact capabilities and limitations
-Engineering & shop drawing support: Detail the support services offered
-Sampling/prototyping capability: Describe the process and timeframes

Production Capacity:-
-Monthly or yearly capacity: Provide specific numbers and types of products
-Factory size and workforce: Specify square footage and number of employees

Quality Control Process:-
-In-house QA/QC: Detail the process and team size
-Third-party inspection (SGS, Intertek, etc.): List specific partners and frequency
-Factory audit report: Specify latest audit dates and results

Language/Communication:-
-English-speaking sales/engineering team: Specify team size and availability

Logistics Support:-
-Export experience: List specific shipping terms offered with details
-In-house packing team: Describe team size and capabilities
-Crating & labeling capabilities: Detail specific services offered

Company Profile or Catalog:-
-PDF or online catalog: Specify format and content details
-Website or GlobalSources profile: Provide specific platform presence

Project References:-
-Hotel brands or commercial clients: Name specific brands and projects
-Photos of past projects: Describe available documentation

Factory Tour or Video:-
-Virtual tour: Specify format and accessibility
-Google Maps verified: Include verification details

Ability to Handle Large-Scale Projects:-
-Multi-phase, multi-location deliveries: Provide specific project examples

After-Sales Support:-
-Warranty: Detail specific coverage and terms
-Spare parts supply: Specify availability and delivery times
-Installation guides: Detail format and languages available

Client Reviews or Ratings:-
-From third-party platforms or references: Include specific ratings and review sources

For each category and subcategory above, provide detailed, specific information. If information is not found, state 'Information not found'. Avoid simple yes/no answers - instead, provide concrete details, numbers, and specific capabilities where available.

Here is the collected information to analyze:

{collected_data_text}"""

                    try:
                        with st.spinner("Analyzing all collected data with OpenAI... This may take a moment."):
                            try:
                                response = client.chat.completions.create(
                                    model="gpt-4o-mini",
                                    messages=[
                                        {"role": "system", "content": "You are a highly skilled AI assistant that extracts and summarizes specific company information. Always maintain the exact format provided in the prompt, including all categories and subcategories with their exact punctuation and indentation."},
                                        {"role": "user", "content": final_summary_prompt}
                                    ],
                                    temperature=0.0
                                )
                                
                                if response and response.choices and len(response.choices) > 0:
                                    analysis_result = response.choices[0].message.content
                                    
                                    # Store the analysis result in session state
                                    st.session_state['analysis_results'][supplier['name']] = analysis_result
                                    st.session_state['analyzed_suppliers'].add(supplier['name'])
                                    
                                    st.success("Deep Dive Analysis Complete!")
                                    st.markdown("---")
                                    st.subheader("📊 Company Information Summary:")
                                    with st.expander("View Detailed Analysis", expanded=True):
                                        st.markdown("```\n" + analysis_result + "\n```")
                                        
                                    # Set to view this supplier's results after analysis
                                    st.session_state['view_supplier_name'] = supplier['name']
                                else:
                                    st.error("Received an empty response from OpenAI. Please try again.")
                                    
                            except Exception as e:
                                st.error(f"An error occurred while calling OpenAI API: {str(e)}")
                                    
                    except Exception as e:
                        st.error(f"An error occurred while processing the results: {str(e)}")
                        
                    if not collected_data_text.strip():
                        st.warning("No data collected to process with OpenAI.")

                st.markdown("---")

            # Reset the analysis state after displaying results
            if 'analyze_all' in st.session_state:
                del st.session_state['analyze_all']
            st.session_state['perform_analysis'] = False
            
            # Rerun to show the results view
            st.rerun()

# --- Supplier Analysis Box in Sidebar ---
st.sidebar.header("Suppliers for Analysis")

if st.session_state['suppliers_for_analysis']:
    # Display count of suppliers and analyzed suppliers
    st.sidebar.info(f"Total suppliers: {len(st.session_state['suppliers_for_analysis'])} | Analyzed: {len(st.session_state['analyzed_suppliers'])}")
    
    # List suppliers in sidebar
    for i, supplier in enumerate(st.session_state['suppliers_for_analysis']):
        col1, col2, col3 = st.sidebar.columns([0.6, 0.2, 0.2])
        with col1:
            st.sidebar.write(f"- {supplier['name']}")
        with col2:
            # Show analysis status indicator
            if supplier['name'] in st.session_state['analyzed_suppliers']:
                st.sidebar.success("✓")
        with col3:
            if st.sidebar.button("×", key=f"remove_btn_{i}"):
                # Remove from analyzed suppliers if present
                if supplier['name'] in st.session_state['analyzed_suppliers']:
                    st.session_state['analyzed_suppliers'].remove(supplier['name'])
                # Remove from analysis results if present
                if supplier['name'] in st.session_state['analysis_results']:
                    del st.session_state['analysis_results'][supplier['name']]
                # Remove from suppliers list
                st.session_state['suppliers_for_analysis'].pop(i)
                st.rerun()

    st.sidebar.markdown("---")
    
    # Add button to analyze all suppliers
    if st.sidebar.button("Analyze All", key="sidebar_analyze_all"):
        st.session_state['perform_analysis'] = True
        st.session_state['analyze_all'] = True
        st.rerun()
else:
    st.sidebar.info("No suppliers added for analysis yet.")
