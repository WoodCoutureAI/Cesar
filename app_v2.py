import streamlit as st
import requests
import re
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from readability import Document
import json
import ast
import random  # Add at the top with other imports
from fpdf import FPDF

# Perplexity API Key (store securely in production)
PERPLEXITY_API_KEY = st.secrets.get("PERPLEXITY_API_KEY", None)

def generate_pdf_from_text(text, company_name="", website_url="", filename="document.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    if company_name:
        pdf.set_font("Arial", 'B', 16)
        sanitized_company_name = company_name.encode('latin-1', 'replace').decode('latin-1')
        pdf.cell(0, 10, sanitized_company_name, 0, 1, 'C')
        pdf.ln(5)
    if website_url:
        pdf.set_font("Arial", 'U', 12)
        pdf.set_text_color(0, 0, 255)
        sanitized_website_url = website_url.encode('latin-1', 'replace').decode('latin-1')
        pdf.write(8, sanitized_website_url)
        pdf.ln(10)
        pdf.set_text_color(0, 0, 0) # Reset color
    pdf.set_font("Arial", size=12)
    # Split text into lines, handling potential long lines and encoding
    for line in text.split('\n'):
        try:
            sanitized_line = line.encode('latin-1', 'replace').decode('latin-1')
            sanitized_line = sanitized_line.replace('**', '') # Remove markdown bolding
            pdf.write(8, sanitized_line)
            pdf.ln()
        except Exception as e:
            st.warning(f"Could not write line to PDF due to encoding issue: {e}")
            try:
                sanitized_line_fallback = line.encode('ascii', 'replace').decode('ascii') # Fallback to ASCII
                sanitized_line_fallback = sanitized_line_fallback.replace('**', '') # Remove markdown bolding
                pdf.write(8, sanitized_line_fallback)
                pdf.ln()
            except Exception as e2:
                st.error(f"Failed to write line with utf-8 fallback: {e2}")
    return bytes(pdf.output(dest='S'))

# --- Utility Functions ---
def get_perplexity_answer(prompt):
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "sonar",  # Changed to sonar model as requested
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }
    response = requests.post("https://api.perplexity.ai/chat/completions", headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        answer = data["choices"][0]["message"]["content"]
        return answer, []  # No citations in this endpoint
    else:
        st.error(f"Perplexity API error: {response.text}")
        return "", []

def get_website_content(website_url, timeout=20, retries=3):
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/114.0.1823.79",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/114.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15"
    ]
    session = requests.Session()
    
    for attempt in range(1, retries + 1):
        try:
            # Rotate User-Agent for each attempt
            headers = {"User-Agent": random.choice(user_agents)}
            session.headers.update(headers)
            response = session.get(website_url, timeout=timeout, verify=True)
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            return response.text
        except requests.exceptions.RequestException as e:
            # st.warning(f"Attempt {attempt}/{retries} failed for {website_url}: {e}") # Suppress UI warning
            if attempt == retries:
                # st.error(f"All {retries} attempts to fetch {website_url} failed.") # Suppress UI error
                pass # Do nothing, just return empty string
    return ""

def extract_main_content(html):
    try:
        doc = Document(html)
        summary_html = doc.summary()
        soup = BeautifulSoup(summary_html, "html.parser")
        content = soup.get_text(separator=" ", strip=True)
        
        # If readability returns very little content, try a more aggressive BeautifulSoup extraction
        if len(content.split()) < 50: # Arbitrary threshold, can be adjusted
            # st.warning("Readability returned limited content, attempting aggressive extraction.")
            soup = BeautifulSoup(html, "html.parser")
            paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
            aggressive_content = " ".join([p.get_text(separator=" ", strip=True) for p in paragraphs])
            if len(aggressive_content.split()) > len(content.split()):
                content = aggressive_content
        return content
    except Exception as e:
        st.warning("Error using readability, falling back to full HTML parsing: " + str(e))
        soup = BeautifulSoup(html, "html.parser")
        paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
        aggressive_content = " ".join([p.get_text(separator=" ", strip=True) for p in paragraphs])
        return aggressive_content

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
    email_regex = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    for email in re.findall(email_regex, html):
        emails.add(email)
    # Phones
    phone_re = re.compile(r'''\+
    \d{1,3}
    (?:[\s\-\(\)]*\d){7,12}
    ''', re.VERBOSE)
    for candidate in phone_re.findall(html):
        digits = re.sub(r"[^\d+]", "", candidate)
        if 8 <= len(digits) <= 15:
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

def scrape_manufacturer_website(website_url):
    with st.spinner(f"Scraping website content from {website_url}..."):
        homepage_html = get_website_content(website_url)
        if not homepage_html:
            return "", [], [], []
        homepage_content = extract_main_content(homepage_html)
        # If homepage_content is still empty after extraction, return early
        if not homepage_content.strip():
            st.warning(f"No meaningful content extracted from homepage of {website_url}")
            return "", [], [], []

        homepage_emails, homepage_phones, homepage_addresses = extract_contact_details(homepage_html)
        keywords = ['about', 'products', 'contact', 'contact us', 'services', 'portfolio', 'get in touch', 'inquiry', 'catalogue', 'company', 'profile', 'overview', 'Products & Services', 'Projects']
        relevant_links = find_relevant_links(homepage_html, website_url, keywords)
        extracted_content = {"Homepage": homepage_content}
        all_emails = set(homepage_emails)
        all_phones = set(homepage_phones)
        all_addresses = set(homepage_addresses)
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
                page_emails, page_phones, page_addresses = extract_contact_details(page_html)
                all_emails.update(page_emails)
                all_phones.update(page_phones)
                all_addresses.update(page_addresses)
        combined_content = ""
        for section, content in extracted_content.items():
            combined_content += f"\n--- {section} ---\n{content}\n"
        return combined_content, list(all_emails), list(all_phones), list(all_addresses)

def generate_manufacturer_summary_from_content(company_name, extracted_content):
    with st.spinner(f"Generating detailed summary for {company_name}..."):
        prompt = f"""
        You are a fine-tuned business research assistant. Based on the following extracted website content from '{company_name}', generate a detailed summary that includes:
        - Company Size (give specific numbers if available, e.g., number of employees, factory size)
        - Years in Business (give specific years or founding date)
        - Types of Products (list specific products)
        - Client Portfolio (list specific clients or projects)
        - Industry Certifications (list certification names and numbers)
        - Product Catalogues (give links or file names if available)
        - Manufacturing Capabilities (give specific numbers, e.g., production capacity)
        - Quality Standards (list standards and details)
        - Export Information (list countries, volumes, or terms)
        - Contact Details (include all emails, phone numbers, and physical addresses)
        For each field, provide specific numbers and details if available. If information is not found, state 'Information not found'. Do not invent details.
        **Do NOT use tables or any tabular format. Do NOT use a summary table. Only use bullet points or paragraphs.**
        Extracted Content:
        {extracted_content}
        Please output the final summary in a clear, professional, and structured manner. Do not use tables, only use bullet points or paragraphs.
        """
        if not PERPLEXITY_API_KEY:
            return "API key not available for summary generation. Please configure your PERPLEXITY_API_KEY."
        try:
            answer, citations = get_perplexity_answer(prompt)
            answer = convert_table_to_bullets(answer)
            return answer
        except Exception as e:
            return f"Could not generate summary due to API error: {e}. Please check your API key or try again later."

def is_aggregator_title(title):
    blacklist_keywords = ['top', 'best', 'guide', 'list', 'review']
    return any(kw.lower() in title.lower() for kw in blacklist_keywords)

def extract_manufacturer_info(company_name):
    linkedin_url = None
    # Use Perplexity to get both LinkedIn and website in one go
    prompt = f"Find the official website and LinkedIn page for the company named '{company_name}'. Return both URLs if available. Prioritize the official website over LinkedIn. Format: Official Website: [URL], LinkedIn: [URL]"
    answer, _ = get_perplexity_answer(prompt)

    linkedin_url = None
    website_url = None
    website_candidates = []

    for line in answer.splitlines():
        # Use regex to find URLs in the line
        url_matches = re.findall(r'https?://(?:www\.)?[^\s/$.?#].[^\s]*', line)

        for url in url_matches:
            if "linkedin.com" in url.lower():
                linkedin_url = url
            elif url.startswith("http") and "linkedin.com" not in url.lower():
                website_candidates.append(url)
    
    # Pick the first website candidate as the official website
    if website_candidates:
        website_url = website_candidates[0]

    return linkedin_url, website_url

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

def parse_markdown_table(md_table):
    """
    Parse a markdown table into a list of dicts with keys: title, link, snippet.
    """
    import re
    lines = [line.strip() for line in md_table.strip().splitlines() if line.strip()]
    if not lines or '|' not in lines[0]:
        return []
    headers = [h.strip().lower() for h in lines[0].split('|') if h.strip()]
    results = []
    for line in lines[2:]:  # skip header and separator
        cols = [c.strip() for c in line.split('|')]
        if len(cols) < 2:
            continue
        entry = {}
        for i, h in enumerate(headers):
            if i < len(cols):
                entry[h] = cols[i]
        # Map to expected keys
        results.append({
            'title': entry.get('title', entry.get('company', '')),
            'link': entry.get('link', entry.get('website', '')),
            'snippet': entry.get('snippet', entry.get('description', ''))
        })
    return results

def parse_search_results(answer):
    import re
    import json
    import ast

    # Try JSON
    try:
        data = json.loads(answer)
        if isinstance(data, list):
            return {'organic': data}
        if isinstance(data, dict) and 'organic' in data:
            return data
    except Exception:
        pass
    # Try Python literal
    try:
        data = ast.literal_eval(answer)
        if isinstance(data, list):
            return {'organic': data}
    except Exception:
        pass
    # Try to extract markdown-style links: [Title](URL)
    results = []
    for match in re.finditer(r'\[([^\]]+)\]\((https?://[^\)]+)\)', answer):
        title, url = match.groups()
        results.append({'title': title, 'link': url, 'snippet': ''})
    # Try to extract markdown table
    if not results and '|' in answer and '---' in answer:
        table_results = parse_markdown_table(answer)
        if table_results:
            return {'organic': table_results}
    # Try to extract bare links with preceding lines as titles
    if not results:
        for line in answer.splitlines():
            url_match = re.search(r'(https?://\S+)', line)
            if url_match:
                url = url_match.group(1)
                title = line.replace(url, '').strip('-:• \n')
                results.append({'title': title or url, 'link': url, 'snippet': ''})
    if results:
        return {'organic': results}
    return None

def convert_table_to_bullets(text):
    """
    If the text contains a markdown table, convert it to bullet points.
    """
    import re
    lines = text.splitlines()
    table_lines = []
    in_table = False
    for line in lines:
        if "|" in line:
            in_table = True
            table_lines.append(line)
        elif in_table and line.strip() == "":
            break
    if not table_lines or len(table_lines) < 3:
        return text  # No table found

    # Parse table
    headers = [h.strip() for h in table_lines[0].split("|") if h.strip()]
    bullets = []
    for row in table_lines[2:]:
        cols = [c.strip() for c in row.split("|")]
        if len(cols) < len(headers):
            continue
        bullet = []
        for h, c in zip(headers, cols):
            if c:
                bullet.append(f"**{h}:** {c}")
        if bullet:
            bullets.append("- " + "; ".join(bullet))
    # Remove table from text and add bullets
    non_table = [line for line in lines if line not in table_lines]
    return "\n".join(non_table) + "\n" + "\n".join(bullets)

def parse_user_prompt_for_search(user_prompt):
    """
    Uses an LLM to parse the user's natural language prompt to extract:
    1. The number of results to find.
    2. A list of effective search queries.
    """
    import re
    import json

    # 1. Extract number of results, default to 5
    max_results = 5
    match = re.search(r'\b(\d+)\b', user_prompt)
    if match:
        num = int(match.group(1))
        if 1 <= num <= 50:  # Cap at 50 for safety
            max_results = num

    # 2. Generate search terms with Perplexity API
    prompt_for_llm = f"""
    You are a search query generation expert. Based on the user's request below, create a JSON array of 5 diverse and effective Google search queries to find relevant manufacturers.
    The queries should be specific and varied to maximize the chances of finding unique, high-quality results.
    Respond ONLY with the JSON array, nothing else.

    User request: "{user_prompt}"

    Example for "Find me 10 bespoke furniture manufacturers in Italy that work with marble":
    ["bespoke furniture manufacturers Italy marble", "luxury custom furniture companies Italy stone work", "high-end joinery workshops Milan Italy", "Italian furniture makers for hospitality projects", "custom wood and marble furniture suppliers Italy"]
    """
    try:
        answer, _ = get_perplexity_answer(prompt_for_llm)
        # The answer should be a JSON string
        search_terms = json.loads(answer)
        if isinstance(search_terms, list) and all(isinstance(term, str) for term in search_terms):
            return max_results, search_terms
    except (json.JSONDecodeError, TypeError):
        # Fallback if the LLM fails to return valid JSON
        pass
    
    # Fallback: use the user's prompt as a single search term
    return max_results, [user_prompt.replace('"', '')]

def is_official_email(email, website_url):
    """Check if the email domain matches the website domain."""
    from urllib.parse import urlparse
    domain = urlparse(website_url).netloc.replace('www.', '').lower()
    email_domain = email.split('@')[-1].lower()
    return domain in email_domain

def is_valid_phone(phone):
    """Filter out placeholder or obviously fake numbers."""
    import re
    digits = re.sub(r"[^\d+]", "", phone)
    # Ignore numbers that are too short or too long
    if not (8 <= len(digits) <= 15):
        return False
    # Ignore numbers with obvious patterns (e.g., ranges, repeated digits, or containing '-')
    if re.match(r".*0{3,}.*", digits):  # e.g., 0000
        return False
    if '-' in phone or 'to' in phone or ',' in phone:
        return False
    return True

def extract_contact_details_from_official_site(base_url, homepage_html):
    """
    Extracts emails and phone numbers from the homepage and direct subpages (contact/about) only.
    Only returns official emails and valid phone numbers.
    """
    emails, phones, addresses = set(), set(), []
    soup = BeautifulSoup(homepage_html, "html.parser")
    # Homepage
    e, p, a = extract_contact_details(homepage_html)
    # Filter emails and phones
    e = [email for email in e if is_official_email(email, base_url)]
    p = [phone for phone in p if is_valid_phone(phone)]
    emails.update(e)
    phones.update(p)
    addresses.extend(a)
    # Try /contact and /about if available
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].lower()
        if any(x in href for x in ["contact", "about"]):
            sub_url = urljoin(base_url, a_tag["href"])
            sub_html = get_website_content(sub_url)
            if sub_html:
                e, p, a = extract_contact_details(sub_html)
                e = [email for email in e if is_official_email(email, base_url)]
                p = [phone for phone in p if is_valid_phone(phone)]
                emails.update(e)
                phones.update(p)
                addresses.extend(a)
    return list(emails), list(phones), list(set(addresses))

def get_contact_info_with_fallback(company_name, website_url, homepage_html):
    """
    Try to get contact info from the official website. If not found, use Perplexity API as a fallback.
    """
    emails, phones, addresses = extract_contact_details_from_official_site(website_url, homepage_html)
    if emails or phones:
        return emails, phones, addresses
    # Fallback: Use Perplexity API to find official contact info
    prompt = (
        f"Find the official email address and phone number for the company '{company_name}' from its website '{website_url}'. "
        "Respond ONLY with a JSON object: {\"emails\": [..], \"phones\": [..], \"addresses\": [..]}. "
        "If not found, use empty lists."
    )
    answer, _ = get_perplexity_answer(prompt)
    import json
    try:
        data = json.loads(answer)
        return data.get("emails", []), data.get("phones", []), data.get("addresses", [])
    except Exception:
        return [], [], []

# --- Streamlit UI and main logic remain unchanged, using only the above functions ---

def run_app():
    # st.title("Wood Couture AI Market Scout")
    st.markdown(
        '<h1 style="color:#b0680c;">Wood Couture AI Market Scout</h1>',
        unsafe_allow_html=True
    )
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
    # Initialize current_search_mode if not set
    if 'current_search_mode' not in st.session_state:
        st.session_state['current_search_mode'] = "General Search"

    # Create two tabs
    search_tab, analysis_tab = st.tabs(["Search", "Analysis Results"])

    with search_tab:
        # Removed update_search_mode function and simplified selectbox state management
        st.session_state['current_search_mode'] = st.selectbox(
            "Select Search Mode",
            ("General Search", "Specific Company Search"),
            key="search_mode_selectbox", # Keep the key
        )

        # --- START OF GENERAL SEARCH BLOCK ---
        if st.session_state['current_search_mode'] == "General Search":
            st.header("General Manufacturer Search")

            user_prompt = st.text_area(
                "Describe the manufacturers you want to find.",
                placeholder="e.g., 'Find me 10 bespoke furniture manufacturers in Vietnam specializing in outdoor furniture'",
                height=100,
                key="general_search_prompt" # Added key
            )
            if st.button("Perform General Search", key="perform_general_search_button"): # Added key
                if not PERPLEXITY_API_KEY:
                    # st.error("Cannot perform search: PERPLEXITY_API_KEY is not set.")
                    pass
                elif not user_prompt.strip():
                    # st.error("Please enter a description of what you're looking for.")
                    pass
                else:
                    with st.spinner("Analyzing your request and preparing search queries..."):
                        max_results, search_terms = parse_user_prompt_for_search(user_prompt)
                        st.info(f"Searching for up to {max_results} companies using {len(search_terms)} search strategies.")

                    st.session_state['last_general_search_results'] = {}
                    st.session_state['last_specific_company_summary'] = None # Clear specific search if general initiated

                    search_count = 0
                    seen_urls = set()
                    seen_names = set()
                    max_attempts = max_results * 10  # Prevent infinite loops
                    attempts = 0
                    term_index = 0

                    while search_count < max_results and attempts < max_attempts:
                        term = search_terms[term_index % len(search_terms)]
                        query = term
                        st.info(f"Performing search: {query}")
                        prompt = (
                            f"You are an API. Perform a search for the query: '{query}'. "
                            "Respond ONLY with a valid JSON array of objects, each with 'title', 'link', and 'snippet' fields. "
                            "Do NOT include any text, markdown, or explanation before or after the JSON. "
                            f"Return at least {max_results * 2} results if possible. "
                            "If you cannot find results, return an empty array []. "
                            "Example: "
                            "[\{\"title\": \"Company A\", \"link\": \"https://companya.com\", \"snippet\": \"Company A is a...\"}, ...]"
                        )
                        answer, _ = get_perplexity_answer(prompt)
                        search_results = parse_search_results(answer)

                        if not search_results or 'organic' not in search_results or not search_results['organic']:
                            # st.warning(f"No valid search results could be extracted for the query: '{query}'.")
                            attempts += 1
                            term_index += 1
                            continue

                        for result in search_results['organic']:
                            if search_count >= max_results:
                                break

                            company_title = result.get('title', '')
                            if is_aggregator_title(company_title):
                                continue
                            website_url_candidate = result.get('link')
                            if not website_url_candidate:
                                continue
                            website_url_candidate = website_url_candidate.strip('". ')
                            # Skip if already seen
                            if website_url_candidate in seen_urls or company_title in seen_names:
                                continue
                            skip_domains = [
                                'made-in-china.com', 'alibaba.com', 'thomasnet.com', 'yellowpages', 'quora.com',
                                'reddit.com', 'facebook.com', 'globalsources.com', 'homedepot.com', 'indiamart.com',
                                'justdial.com', 'yelp.com', 'amazon.com', 'ebay.com', 'wikipedia.org', 'tripadvisor.com',
                                'pinterest.com', 'instagram.com', 'youtube.com', 'twitter.com', 'm.alibaba.com',
                                'blog', 'review', 'list', 'top', 'best'
                            ]
                            if any(domain in website_url_candidate for domain in skip_domains):
                                continue

                            homepage_html = get_website_content(website_url_candidate)
                            if not homepage_html:
                                continue

                            # Extract contact info from official website (homepage + direct subpages)
                            all_emails, all_phones, all_addresses = get_contact_info_with_fallback(company_title, website_url_candidate, homepage_html)
                            # Do NOT filter out companies without email/phone; always include

                            linkedin_url, _ = extract_manufacturer_info(company_title)
                            if company_title not in st.session_state['last_general_search_results']:
                                with st.spinner(f"Scraping and summarizing {company_title} website..."):
                                    combined_content, _, _, _ = scrape_manufacturer_website(website_url_candidate)
                                    if not combined_content:
                                        # st.warning(f"Failed to extract meaningful content from {website_url_candidate} for {company_title}")
                                        continue
                                    summary = generate_manufacturer_summary_from_content(company_title, combined_content)
                                st.session_state['last_general_search_results'][company_title] = {
                                    "website_url": website_url_candidate,
                                    "linkedin_url": linkedin_url,
                                    "summary": summary,
                                    "emails": all_emails,
                                    "phones": all_phones,
                                    "addresses": all_addresses,
                                    "extracted_content": combined_content,
                                    "name": company_title
                                }
                                seen_urls.add(website_url_candidate)
                                seen_names.add(company_title)
                                search_count += 1
                                if search_count >= max_results:
                                    break

                        attempts += 1
                        term_index += 1

                    if not st.session_state['last_general_search_results']:
                        st.error("No manufacturers found. Please try a different prompt.")
                    else:
                        st.info(f"Search complete. Found {search_count} companies.") # Indicate search completion with count
            
            # ALWAYS DISPLAY RESULTS IF THEY EXIST IN SESSION STATE
            if st.session_state['last_general_search_results']:
                st.subheader("Found Manufacturers:")
                # Convert to list for ordering
                results_list = list(st.session_state['last_general_search_results'].items())
                for idx, (company_name, details) in enumerate(results_list):
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
                    
                    # Download button for general search results
                    pdf_filename = f"{company_name.replace(' ', '_')}_Summary.pdf"
                    pdf_content = generate_pdf_from_text(details['summary'], company_name, details['website_url'], pdf_filename)
                    st.download_button(
                        label="Download Summary as PDF",
                        data=pdf_content,
                        file_name=pdf_filename,
                        mime="application/pdf",
                        key=f"download_pdf_general_{company_name.replace(' ', '_')}"
                    )
                    st.markdown("---")

                # --- Add the 'Get Different Results' button ---
                get_diff_results_btn = st.button("Get Different Results 🔄", key="get_different_results_btn_general")
                if get_diff_results_btn:
                    # Keep all previous results, only add new unique results below
                    user_prompt_for_diff_search = st.session_state.get('last_user_prompt', '')
                    if not user_prompt_for_diff_search:
                        st.warning("No previous prompt found. Please enter a new search.")
                    else:
                        with st.spinner("Generating different search queries and results..."):
                            import random
                            _, search_terms = parse_user_prompt_for_search(user_prompt_for_diff_search)
                            random.shuffle(search_terms)
                            new_results = dict(st.session_state['last_general_search_results'])
                            seen_urls = set(details['website_url'] for details in new_results.values())
                            seen_names = set(new_results.keys())
                            for s in st.session_state['suppliers_for_analysis']:
                                seen_urls.add(s['website_url'])
                                seen_names.add(s['name'])
                            search_count = 0
                            attempts = 0
                            term_index = 0
                            max_attempts = 50  # Reasonable cap to avoid infinite loop
                            while search_count < 3 and attempts < max_attempts:  # Fetch 3 new results per click
                                term = search_terms[term_index % len(search_terms)]
                                query = term + f" unique {random.randint(1000,9999)}"
                                prompt = (
                                    f"You are an API. Perform a search for the query: '{query}'. "
                                    "Respond ONLY with a valid JSON array of objects, each with 'title', 'link', and 'snippet' fields. "
                                    "Do NOT include any text, markdown, or explanation before or after the JSON. "
                                    f"Return at least 3 results if possible. "
                                    "If you cannot find results, return an empty array []. "
                                    "Example: "
                                    "[\{\"title\": \"Company A\", \"link\": \"https://companya.com\", \"snippet\": \"Company A is a...\"}, ...]"
                                )
                                try:
                                    answer, _ = get_perplexity_answer(prompt)
                                except Exception as e:
                                    attempts += 1
                                    term_index += 1
                                    continue
                                # If API returns an error page, skip
                                if not answer or '502 Bad Gateway' in answer or '<html>' in answer:
                                    attempts += 1
                                    term_index += 1
                                    continue
                                search_results = parse_search_results(answer)
                                if not search_results or 'organic' not in search_results or not search_results['organic']:
                                    attempts += 1
                                    term_index += 1
                                    continue
                                for result in search_results['organic']:
                                    if search_count >= 3:
                                        break
                                    company_title = result.get('title', '')
                                    website_url_candidate = result.get('link')
                                    if not website_url_candidate:
                                        continue
                                    website_url_candidate = website_url_candidate.strip('". ')
                                    # Skip if already seen (by name or URL)
                                    if website_url_candidate in seen_urls or company_title in seen_names:
                                        continue
                                    # --- Apply same filtering as initial search ---
                                    if is_aggregator_title(company_title):
                                        continue
                                    skip_domains = [
                                        'made-in-china.com', 'alibaba.com', 'thomasnet.com', 'yellowpages', 'quora.com',
                                        'reddit.com', 'facebook.com', 'globalsources.com', 'homedepot.com', 'indiamart.com',
                                        'justdial.com', 'yelp.com', 'amazon.com', 'ebay.com', 'wikipedia.org', 'tripadvisor.com',
                                        'pinterest.com', 'instagram.com', 'youtube.com', 'twitter.com', 'm.alibaba.com',
                                        'blog', 'review', 'list', 'top', 'best'
                                    ]
                                    if any(domain in website_url_candidate for domain in skip_domains):
                                        continue
                                    # --- FULL: Do full scraping and summary as in initial search ---
                                    homepage_html = get_website_content(website_url_candidate)
                                    if not homepage_html:
                                        continue
                                    all_emails, all_phones, all_addresses = get_contact_info_with_fallback(company_title, website_url_candidate, homepage_html)
                                    linkedin_url, _ = extract_manufacturer_info(company_title)
                                    combined_content, _, _, _ = scrape_manufacturer_website(website_url_candidate)
                                    if not combined_content:
                                        continue
                                    summary = generate_manufacturer_summary_from_content(company_title, combined_content)
                                    new_results[company_title] = {
                                        "website_url": website_url_candidate,
                                        "linkedin_url": linkedin_url,
                                        "summary": summary,
                                        "emails": all_emails,
                                        "phones": all_phones,
                                        "addresses": all_addresses,
                                        "extracted_content": combined_content,
                                        "name": company_title
                                    }
                                    seen_urls.add(website_url_candidate)
                                    seen_names.add(company_title)
                                    search_count += 1
                                    if search_count >= 3:
                                        break
                            attempts += 1
                            term_index += 1
                        st.session_state['last_general_search_results'] = new_results
                        st.experimental_rerun()

    # --- END OF GENERAL SEARCH BLOCK ---

    # Refactored Specific Company Search UI into a function to isolate rendering issues
    def _render_specific_company_search_ui():
        # Use st.empty() to create a dedicated slot for the UI, to aggressively force rendering
        ui_placeholder = st.empty()
        with ui_placeholder.container():
            st.header("Specific Company Search")
            specific_company_input = st.text_input("Enter the name of the specific company/manufacturer",
                                                   value=st.session_state['last_specific_company_name_input'],
                                                   key="specific_company_name_input")

            if st.button("Search Specific Company", key="search_specific_company_btn"):
                if not PERPLEXITY_API_KEY:
                    # st.error("Cannot perform search: PERPLEXITY_API_KEY is not set.")
                    pass
                elif specific_company_input:
                    st.session_state['last_specific_company_name_input'] = specific_company_input
                    st.session_state['last_general_search_results'] = {}
                    st.session_state['last_specific_company_summary'] = None
                    # Clear suppliers for analysis when a specific company search is initiated
                    st.session_state['suppliers_for_analysis'] = [] 
                    st.session_state['analysis_results'] = {}
                    st.session_state['analyzed_suppliers'] = set()

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
                                # st.error("Failed to extract content from the website.")
                                pass
                                st.session_state['last_specific_company_summary'] = None
                            else:
                                summary = generate_manufacturer_summary_from_content(specific_company_input, combined_content)

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
                    # st.error("No company provided. Please enter a company name.")
                    pass

            # ALWAYS DISPLAY SPECIFIC COMPANY RESULTS IF THEY EXIST IN SESSION STATE
            if st.session_state['last_specific_company_summary'] and st.session_state['current_search_mode'] == "Specific Company Search":
                details = st.session_state['last_specific_company_summary']
                st.subheader(f"Manufacturer: {details['name']}")
                st.write(f"**Website:** {details['website_url']}")
                if details['linkedin_url']:
                    st.write(f"**LinkedIn:** {details['linkedin_url']}")
                st.markdown(details['summary'])

                # Removed the 'Add to Analysis Box' button for Specific Company Search
                # if st.button(f"Add {details['name']} to Analysis Box", key=f"add_specific_company_to_analysis_{details['name'].replace(' ', '_')}"):
                #     supplier_data = details
                #     if not any(s['website_url'] == supplier_data['website_url'] for s in st.session_state['suppliers_for_analysis']):
                #         st.session_state['suppliers_for_analysis'].append(supplier_data)
                #         st.success(f"{details['name']} added to analysis box!")
                #     else:
                #         st.info(f"{details['name']} is already in the analysis box.")
                
                pdf_filename = f"{details['name'].replace(' ', '_')}_Summary.pdf"
                pdf_content = generate_pdf_from_text(details['summary'], details['name'], details['website_url'], pdf_filename)
                st.download_button(
                    label="Download Summary as PDF",
                    data=pdf_content,
                    file_name=pdf_filename,
                    mime="application/pdf",
                    key=f"download_pdf_specific_summary_{details['name'].replace(' ', '_')}"
                )
                st.markdown("---")
            elif specific_company_input and not st.session_state['last_specific_company_summary'] and st.session_state['current_search_mode'] == "Specific Company Search":
                pass

    if st.session_state['current_search_mode'] == "Specific Company Search":
        _render_specific_company_search_ui()

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
                st.write("**Company Name**")
            with col2:
                st.write("**Analyze**")
            with col3:
                st.write("**Status**")
            with col4:
                st.write("**Remove**")
            
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
                        # Remove bold formatting before displaying
                        analysis_result = st.session_state['analysis_results'][supplier_name]
                        analysis_result = analysis_result.replace('**', '')
                        # Try to detect a markdown table and display as a DataFrame/table
                        import re
                        import pandas as pd
                        table_match = re.search(r'\|\s*Category\s*\|\s*Summary\s*\|', analysis_result)
                        if table_match:
                            # Extract table rows
                            lines = [line for line in analysis_result.splitlines() if '|' in line and not line.strip().startswith('|-')]
                            rows = [line.strip().strip('|').split('|') for line in lines]
                            rows = [[cell.strip() for cell in row] for row in rows]
                            if rows and len(rows[0]) == 2:
                                # Remove header if present
                                if rows[0][0].lower() == 'category' and rows[0][1].lower() == 'summary':
                                    rows = rows[1:]
                                df = pd.DataFrame(rows, columns=["Category", "Summary"])
                                st.table(df)
                            else:
                                st.markdown(f"""
    ```
    {analysis_result}
    ```
    """)
                        else:
                            st.markdown(f"""
    ```
    {analysis_result}
    ```
    """)
                    
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
                        # Define search filters (these are no longer used for Perplexity search, but kept for context)
                        search_filters = [
                            "Location",
                            "Minimum Order Quantity (MOQ)",
                            "Lead Time",
                            "Certifications & Standards",
                            "Past Project Experience",
                            "Customization Capability",
                            "Production Capacity",
                            "Quality Control Process",
                            "Language/Communication",
                            "Logistics Support",
                            "Company Profile or Catalog",
                            "Project References",
                            "Factory Tour or Video",
                            "Ability to Handle Large-Scale Projects",
                            "After-Sales Support",
                            "Client Reviews or Ratings"
                        ]
                        num_filters = len(search_filters)

                        for i, filter_query in enumerate(search_filters):
                            query = f"{supplier['name']} {filter_query}"
                            relevant_snippets = []
                            relevant_links = []
                            scraped_content = ""

                            # Use Perplexity for search results
                            prompt = (
                                f"You are an API. Perform a search for the query: '{query}'. "
                                "Respond ONLY with a valid JSON array of objects, each with 'title', 'link', and 'snippet' fields. "
                                "Do NOT include any text, markdown, or explanation before or after the JSON. "
                                "Example: "
                                "[{\"title\": \"Company A\", \"link\": \"https://companya.com\", \"snippet\": \"Company A is a...\"}, ...]"
                            )
                            answer, _ = get_perplexity_answer(prompt)
                            search_results = parse_search_results(answer)
                            if not search_results or 'organic' not in search_results:
                                st.warning("No valid search results could be extracted. Raw response shown below for debugging:")
                                #st.code(answer)
                                continue
                            
                            if search_results and 'organic' in search_results:
                                # Collect limited snippets and links
                                for result in search_results['organic'][:3]: # Limit snippets to 3
                                    if result.get('snippet'):
                                        relevant_snippets.append(result.get('snippet', '')[:200])  # Limit snippet length
                                    relevant_links.append(result.get('link', ''))

                                # Strategic Content Extraction (Scraping Top Links)
                                scraped_count = 0
                                for link in relevant_links:
                                    if scraped_count >= 3: # Limit scraping to 3 pages
                                        break
                                    if any(link.endswith(ext) for ext in ['.pdf', '.doc', '.docx', '.jpg', '.png', '.mp4']):
                                        continue
                                    
                                    content = get_website_content(link) # Use get_website_content
                                    if content:
                                        scraped_content += f"\n\n--- Content from {link} ---\n{content}"
                                        scraped_count += 1
                                        # No need for time.sleep(SCRAPE_DELAY) as get_website_content handles retries

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
                        final_summary_prompt = f"""Analyze the following information about '{supplier['name']}' and provide a concise summary for each category. If information is not found, state 'Information not found'.

    Format the response using the following categories:
    1. Location & Logistics
    2. MOQ & Lead Time
    3. Certifications
    4. Project Experience
    5. Manufacturing Capabilities
    6. Quality Control
    7. Communication & Support
    8. Company Profile
    9. References & Reviews

    Keep responses brief but informative.
    {collected_data_text}"""
                        # Create the final prompt with detailed format
    #                     final_summary_prompt = f"""Based on the collected information about the company '{supplier['name']}', 
    # provide a detailed analysis in the following specific format. For each item, provide specific details instead of just yes/no answers. Include numbers, descriptions, and specific capabilities where available:

    # Location / Region
    # Specify the exact city, district, and describe the proximity to major ports or logistics hubs in detail.

    # Minimum Order Quantity (MOQ):-
    # -Per item: Specify exact minimum quantities for different product categories
    # -Per project or container: Specify container or project-based minimums with details

    # Lead Time:-
    # -Sampling lead time: Specify exact timeframes in days/weeks
    # -Production lead time: Specify timeframes for different order sizes
    # -Shipping readiness: Detail preparation and shipping times

    # Certifications & Standards:-
    # -ISO, FSC, CE, UL, RoHS: List specific certification numbers and validity dates if available
    # -Fire-rating compliance (BS5852, CAL117): Detail specific compliance levels and test results
    # -Sustainability/eco-certifications: List specific certifications with details

    # Past Project Experience:-
    # -Hospitality (hotels, resorts): Name specific projects, sizes, and locations
    # -Residential: Detail types and scales of residential projects

    # Customization Capability:-
    # -OEM vs ODM: Specify exact capabilities and limitations
    # -Engineering & shop drawing support: Detail the support services offered
    # -Sampling/prototyping capability: Describe the process and timeframes

    # Production Capacity:-
    # -Monthly or yearly capacity: Provide specific numbers and types of products
    # -Factory size and workforce: Specify square footage and number of employees

    # Quality Control Process:-
    # -In-house QA/QC: Detail the process and team size
    # -Third-party inspection (SGS, Intertek, etc.): List specific partners and frequency
    # -Factory audit report: Specify latest audit dates and results

    # Language/Communication:-
    # -English-speaking sales/engineering team: Specify team size and availability

    # Logistics Support:-
    # -Export experience: List specific shipping terms offered with details
    # -In-house packing team: Describe team size and capabilities
    # -Crating & labeling capabilities: Detail specific services offered

    # Company Profile or Catalog:-
    # -PDF or online catalog: Specify format and content details
    # -Website or GlobalSources profile: Provide specific platform presence

    # Project References:-
    # -Hotel brands or commercial clients: Name specific brands and projects
    # -Photos of past projects: Describe available documentation

    # Factory Tour or Video:-
    # -Virtual tour: Specify format and accessibility
    # -Google Maps verified: Include verification details

    # Ability to Handle Large-Scale Projects:-
    # -Multi-phase, multi-location deliveries: Provide specific project examples

    # After-Sales Support:-
    # -Warranty: Detail specific coverage and terms
    # -Spare parts supply: Specify availability and delivery times
    # -Installation guides: Detail format and languages available

    # Client Reviews or Ratings:-
    # -From third-party platforms or references: Include specific ratings and review sources

    # For each category and subcategory above, provide detailed, specific information. If information is not found, state 'Information not found'. Avoid simple yes/no answers - instead, provide concrete details, numbers, and specific capabilities where available.

    # Here is the collected information to analyze:

    # {collected_data_text}"""

                        try:
                            with st.spinner("Analyzing all collected data with PerplexityAI... This may take a moment."):
                                try:
                                    answer, citations = get_perplexity_answer(final_summary_prompt)
                                    st.markdown(answer)
                                    if citations:
                                        st.markdown("**Sources:**")
                                        for c in citations:
                                            st.markdown(f"- [{c.get('title', c.get('url', 'Source'))}]({c.get('url', '')})")
                                        
                                    # Store the analysis result in session state
                                    st.session_state['analysis_results'][supplier['name']] = answer
                                    st.session_state['analyzed_suppliers'].add(supplier['name'])
                                    
                                    st.success("Deep Dive Analysis Complete!")
                                    st.markdown("---")
                                    with st.expander("View Detailed Analysis", expanded=True):
                                        # Remove bold formatting before displaying
                                        clean_answer = answer.replace('**', '')
                                        # Try to detect a markdown table and display as a DataFrame/table
                                        import re
                                        import pandas as pd
                                        table_match = re.search(r'\|\s*Category\s*\|\s*Summary\s*\|', clean_answer)
                                        if table_match:
                                            lines = [line for line in clean_answer.splitlines() if '|' in line and not line.strip().startswith('|-')]
                                            rows = [line.strip().strip('|').split('|') for line in lines]
                                            rows = [[cell.strip() for cell in row] for row in rows]
                                            if rows and len(rows[0]) == 2:
                                                if rows[0][0].lower() == 'category' and rows[0][1].lower() == 'summary':
                                                    rows = rows[1:]
                                                df = pd.DataFrame(rows, columns=["Category", "Summary"])
                                                st.table(df)
                                            else:
                                                st.markdown(f"""
    ```
    {clean_answer}
    ```
    """)
                                        else:
                                            st.markdown(f"""
    ```
    {clean_answer}
    ```
    """)
                                            
                                    # Set to view this supplier's results after analysis
                                    st.session_state['view_supplier_name'] = supplier['name']
                                    
                                    # Download button for deep dive analysis results
                                    pdf_filename = f"{supplier['name'].replace(' ', '_')}_Deep_Analysis.pdf"
                                    pdf_content = generate_pdf_from_text(answer, supplier['name'], supplier['website_url'], pdf_filename)
                                    st.download_button(
                                        label="Download Deep Analysis as PDF",
                                        data=pdf_content,
                                        file_name=pdf_filename,
                                        mime="application/pdf",
                                        key=f"download_pdf_deep_analysis_{supplier['name'].replace(' ', '_')}"
                                    )
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

    # --- Save the last user prompt for re-use ---
    if st.session_state['current_search_mode'] == "General Search":
        if 'user_prompt' in locals() and user_prompt.strip():
            st.session_state['last_user_prompt'] = user_prompt.strip()
    elif st.session_state['current_search_mode'] == "Specific Company Search":
        if 'specific_company_input' in locals() and specific_company_input.strip(): # Changed to specific_company_input
            st.session_state['last_user_prompt'] = specific_company_input.strip()

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


if __name__ == "__main__":
    run_app()
