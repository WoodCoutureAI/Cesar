import streamlit as st
import requests
import re
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from readability import Document
import openai

# Set your API keys here (or better, via Streamlit secrets)
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
        st.warning("Error using readability: " + str(e))
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator=" ", strip=True)

def extract_contact_details(html):
    emails = set()
    phones = set()
    addresses_found = []

    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=' ', strip=True)  # Extract full text for regex-based address matching

    # Look for mailto links
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("mailto:"):
            email = href.split("mailto:")[1].split("?")[0].strip()
            if email:
                emails.add(email)

    # Regex to find emails in the raw HTML
    email_regex = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    found_emails = re.findall(email_regex, html)
    for email in found_emails:
        emails.add(email)

    # Extract phone numbers
    phone_regex = r'(\+?\d[\d\s\-().]{7,}\d)'
    phone_matches = re.findall(phone_regex, html)
    for phone in phone_matches:
        if len(re.sub(r'[\s\-().]+', '', phone)) >= 7:
            phones.add(phone.strip())

    # Address patterns
    address_regex_patterns = [
        r'\d{1,5}\s(?:[A-Za-z]+\s){1,4}(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Dr|Boulevard|Blvd|Place|Pl|Court|Ct)\.?,?\s*[A-Za-z\s]*,?\s*[A-Z]{2,3}\s*\d{5}(?:-\d{4})?',  # Street, City, State/Zip
        r'(?:P\.O\.\s*Box|PO\s*Box)\s*\d+',  # P.O. Box
        r'[A-Za-z\s]+(?:,\s*[A-Za-z\s]+){1,3}(?:,\s*[A-Z]{2,3})?(?:,\s*\d{4,6})?'  # City, State, Country, Postal Code
    ]

    # Extract addresses from plain text
    for pattern in address_regex_patterns:
        addresses_found.extend(re.findall(pattern, text, re.IGNORECASE))

    addresses = list(set(addr.strip() for addr in addresses_found if addr.strip()))

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
# scrape_manufacturer_website: Scrape homepage and related pages (About, Products, Contact, etc.) and combine content.
def scrape_manufacturer_website(website_url):
    with st.spinner(f"Scraping website content from {website_url}..."):
        homepage_html = get_website_content(website_url)
        if not homepage_html:
            return "", [], []

        homepage_content = extract_main_content(homepage_html)
        # Also extract emails from the homepage HTML
        homepage_emails, homepage_phones, homepage_addresses = extract_contact_details(homepage_html) # Changed this line

        keywords = ['about', 'products', 'contact', 'contact us', 'services', 'portfolio', 'get in touch', 'inquiry', 'catalogue', 'company', 'profile', 'overview', 'Products & Services', 'Projects']
        relevant_links = find_relevant_links(homepage_html, website_url, keywords)
        extracted_content = {"Homepage": homepage_content}

        all_emails = set(homepage_emails)
        all_phones = set(homepage_phones)
        all_addresses = set(homepage_addresses) # Added this line

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
                page_emails, page_phones, page_addresses = extract_contact_details(page_html) # Changed this line
                all_emails.update(page_emails)
                all_phones.update(page_phones)
                all_addresses.update(page_addresses) # Added this line

        combined_content = ""
        for section, content in extracted_content.items():
            combined_content += f"\n--- {section} ---\n{content}\n"
        return combined_content, list(all_emails), list(all_phones), list(all_addresses) # Changed this line
    
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
        - Contact Details (ensure to include email addresses, all phone numbers that are displayed only in the front page, and physical addresses which is a must)

        Use the information provided only in the text below and do not add any invented details.
        
        Extracted Content:
        {extracted_content}

        Please output the final summary in a clear, professional, and structured manner.
        """
        
        if not OPENAI_API_KEY:
            return "API key not available for summary generation. Please configure your OPENAI_API_KEY."
            
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=750,
                temperature=0.7,
            )
            summary = response['choices'][0]['message']['content'].strip()
            return summary
        except Exception as e:
            return "Could not generate summary due to API error. Please check your API key or try again later."

# Is_aggregator_title: Check if a title indicates an aggregator page.
def is_aggregator_title(title):
    blacklist_keywords = ['top', 'best', 'guide', 'list', 'review']
    return any(kw.lower() in title.lower() for kw in blacklist_keywords)

# Extract_manufacturer_info: Extract basic details (LinkedIn and official website) filtering out aggregator pages.
def extract_manufacturer_info(company_name):
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

# Streamlit UI 

st.title("Wood Couture AI Market Scout")
st.markdown("""
I can help you search for and analyze bespoke furniture related manufacturer companies worldwide.
You can perform either a general search by country or search for a specific company.
""")

# Select search mode using a radio button
search_mode = st.radio("Select Search Mode", ("General Search", "Specific Company Search"))

if search_mode == "General Search":
    st.header("General Manufacturer Search")
    # Supplier category buttons
    st.markdown("Choose Supplier Category")
    col1, col2, col3, col4 = st.columns(4)
    # Initialize default
    if 'search_terms' not in st.session_state:
        st.session_state['search_terms'] = [
            "Luxury wood furniture manufacturer",
            "Premium wood furniture manufacturing",
            "Custom wood furniture manufacturer"
        ]
    with col1:
        if st.button("Bespoke Furniture Supplier"):
            st.session_state['search_terms'] = [
                "Luxury wood furniture manufacturer",
                "Premium wood furniture manufacturing",
                "Custom wood furniture manufacturer"
            ]
    with col2:
        if st.button("Lighting Supplier"):
            st.session_state['search_terms'] = [
                "Lighting supplier",
                "Lighting manufacturer",
                "Custom lighting manufacturer"
            ]
    with col3:
        if st.button("Drapery Supplier"):
            st.session_state['search_terms'] = [
                "Drapery supplier",
                "Drapery manufacturer",
                "Custom drapery manufacturer"
            ]
    with col4:
        if st.button("Outdoor Furniture Supplier"):
            st.session_state['search_terms'] = [
                "Outdoor furniture manufacturer",
                "Patio furniture manufacturer",
                "Custom outdoor furniture supplier"
            ]

    country = st.text_input("Enter the country")
    requirements = st.text_input("Enter any specific requirements (optional)")
    max_results = st.number_input("How many companies do you want to display?", min_value=1, max_value=50, value=20)
    
    if st.button("Perform General Search"):
        offset = 0  
        manufacturers = {}
        search_terms = st.session_state['search_terms']
        for term in search_terms:
            query = f"{term} {requirements} in {country}".strip()
            st.info(f"Performing search: {query}")
            search_results = google_search(query, offset=offset)
            if not search_results or 'organic' not in search_results:
                continue
            for result in search_results['organic']:
                company_title = result.get('title', '')
                if is_aggregator_title(company_title):
                    continue
                website_url = result.get('link')
                if website_url and any(excl in website_url for excl in [
                    "alibaba.com", "thomasnet.com", "yellowpages", "quora.com",
                    "made-in-china.com", "reddit.com", "facebook.com", "globalsources.com",
                    "homedepot.com", "indiamart.com", "justdial.com", "yelp.com","amazon.com",
                    "ebay.com", "wikipedia.org", "tripadvisor.com"
                ]):
                    continue
                if company_title not in manufacturers:
                    linkedin_url, official_website = extract_manufacturer_info(company_title)
                    if not official_website:
                        continue
                    manufacturers[company_title] = {
                        "website_url": official_website,
                        "linkedin_url": linkedin_url
                    }
                if len(manufacturers) >= max_results:
                    break
            if len(manufacturers) >= max_results:
                break

        if not manufacturers:
            st.error("No manufacturers found. Please try different search parameters.")
        else:
            for company_name, details in manufacturers.items():
                st.subheader(f"Manufacturer: {company_name}")
                st.write(f"Website: {details['website_url']}")
                st.write(f"LinkedIn: {details['linkedin_url']}")
                with st.spinner(f"Scraping website for {company_name}..."):
                    extracted_content = scrape_manufacturer_website(details['website_url'])
                if not extracted_content:
                    st.warning(f"Failed to extract content from {details['website_url']}")
                    continue
                with st.spinner("Generating summary..."):
                    summary = generate_manufacturer_summary_from_content(company_name, extracted_content)
                st.markdown(summary)
                st.markdown("---")

elif search_mode == "Specific Company Search":
    st.header("Specific Company Search")
    specific_company = st.text_input("Enter the name of the specific company/manufacturer")
    
    if st.button("Search Specific Company"):
        if specific_company:
            linkedin_url, website_url = extract_manufacturer_info(specific_company)
            if not website_url:
                st.error("Could not find a valid website for the specified company.")
            else:
                st.write(f"Website: {website_url}")
                st.write(f"LinkedIn: {linkedin_url}")
                with st.spinner("Scraping website..."):
                    extracted_content = scrape_manufacturer_website(website_url)
                if not extracted_content:
                    st.error("Failed to extract content from the website.")
                else:
                    location = None
                    if linkedin_url:
                        location = extract_linkedin_details(linkedin_url)
                    with st.spinner("Generating summary..."):
                        summary = generate_manufacturer_summary_from_content(specific_company, extracted_content)
                    st.subheader(f"Manufacturer: {specific_company}")
                    #st.write(f"**Phone:** {phone}")
                    #st.write(f"Location: {location}")
                    st.markdown(summary)
                    st.markdown("---")
        else:
            st.error("No company provided. Please enter a company name.")
