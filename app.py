import streamlit as st
import requests
import re
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from readability import Document
import json
import ast
import random
from fpdf import FPDF
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import pandas as pd

# Perplexity API Key (store securely in production)
PERPLEXITY_API_KEY = st.secrets.get("PERPLEXITY_API_KEY", None)

# --- CONFIGURATION ---
MAX_WORKERS = 5  # Number of parallel threads for scraping

# --- Robust JSON Extraction Function ---
def extract_json_from_response(text):
    """
    Finds and parses a JSON array from a string, even if it's embedded in other text.
    """
    # Look for a JSON array embedded in the text
    match = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Fallback for simple cases
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

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
        pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", size=12)
    for line in text.split('\n'):
        try:
            sanitized_line = line.encode('latin-1', 'replace').decode('latin-1').replace('**', '')
            pdf.write(8, sanitized_line)
            pdf.ln()
        except Exception:
            try:
                sanitized_line_fallback = line.encode('ascii', 'replace').decode('ascii').replace('**', '')
                pdf.write(8, sanitized_line_fallback)
                pdf.ln()
            except Exception as e2:
                st.error(f"Failed to write line to PDF: {e2}")
    return bytes(pdf.output(dest='S'))

def get_perplexity_answer(prompt):
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "sonar",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }
    try:
        response = requests.post("https://api.perplexity.ai/chat/completions", headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"], []
    except requests.exceptions.RequestException as e:
        st.error(f"Perplexity API error: {e}")
        return f"Error communicating with API: {e}", []

def get_website_content(website_url, timeout=20):
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/114.0.1823.79",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/114.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15"
    ]
    session = requests.Session()
    session.headers.update({"User-Agent": random.choice(user_agents)})
    try:
        response = session.get(website_url, timeout=timeout, verify=True)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException:
        return ""

def extract_main_content(html):
    try:
        doc = Document(html)
        summary_html = doc.summary()
        soup = BeautifulSoup(summary_html, "html.parser")
        content = soup.get_text(separator=" ", strip=True)
        
        if len(content.split()) < 50:
            soup = BeautifulSoup(html, "html.parser")
            paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
            aggressive_content = " ".join([p.get_text(separator=" ", strip=True) for p in paragraphs])
            if len(aggressive_content.split()) > len(content.split()):
                content = aggressive_content
        return content
    except Exception as e:
        soup = BeautifulSoup(html, "html.parser")
        paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
        return " ".join([p.get_text(separator=" ", strip=True) for p in paragraphs])

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
        r'\d{1,5}\s(?:[A-Za-z]+\s){1,4}(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Dr|Boulevard|Blvd|Place|Pl|Court|Ct)\.?,?\s*[A-Za-z\s]*,?\s*[A-Z]{2,3}\s*\d{5}(?:-\d{4})?',
        r'(?:P\.O\.\s*Box|PO\s*Box)\s*\d+',
    ]
    for pat in address_patterns:
        for match in re.findall(pat, text, re.IGNORECASE):
            addresses_found.append(match.strip())
    
    return list(emails), list(phones), addresses_found

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

def scrape_manufacturer_website_parallel(website_url):
    """Optimized version for parallel processing"""
    homepage_html = get_website_content(website_url)
    if not homepage_html:
        return "", [], [], []
    
    homepage_content = extract_main_content(homepage_html)
    if not homepage_content.strip():
        return "", [], [], []

    homepage_emails, homepage_phones, homepage_addresses = extract_contact_details(homepage_html)
    keywords = ['about', 'products', 'contact', 'services', 'portfolio', 'company']
    relevant_links = find_relevant_links(homepage_html, website_url, keywords)
    
    extracted_content = {"Homepage": homepage_content}
    all_emails = set(homepage_emails)
    all_phones = set(homepage_phones)
    all_addresses = set(homepage_addresses)
    
    # Parallel processing of additional pages
    def scrape_page(link_data):
        key, link = link_data
        page_html = get_website_content(link)
        if page_html:
            page_content = extract_main_content(page_html)
            page_emails, page_phones, page_addresses = extract_contact_details(page_html)
            return key, page_content, page_emails, page_phones, page_addresses
        return key, "", [], [], []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(scrape_page, (key, link)) for key, link in relevant_links.items()]
        for future in as_completed(futures):
            key, page_content, page_emails, page_phones, page_addresses = future.result()
            if page_content:
                extracted_content[key.capitalize()] = page_content
                all_emails.update(page_emails)
                all_phones.update(page_phones)
                all_addresses.update(page_addresses)
    
    combined_content = ""
    for section, content in extracted_content.items():
        combined_content += f"\n--- {section} ---\n{content}\n"
    
    return combined_content, list(all_emails), list(all_phones), list(all_addresses)

def generate_manufacturer_summary_from_content(company_name, extracted_content):
    prompt = f"""
    You are a fine-tuned business research assistant. Based on the following extracted website content from '{company_name}', generate a detailed summary that includes:
    - Company Size (give specific numbers if available, e.g., number of employees, factory size)
    - Years in Business (give specific years or founding date)
    - Types of Products (list specific products)
    - Client Portfolio (list specific clients or projects)
    - Industry Certifications (list certification names and numbers)
    - Manufacturing Capabilities (give specific numbers, e.g., production capacity)
    - Quality Standards (list standards and details)
    - Contact Details (include all emails, phone numbers, and physical addresses)
    For each field, provide specific numbers and details if available. If information is not found, state 'Information not found'. Do not invent details.
    **Do NOT use tables or any tabular format. Do NOT use a summary table. Only use bullet points or paragraphs.**
    Extracted Content:
    {extracted_content[:12000]}
    Please output the final summary in a clear, professional, and structured manner. Do not use tables, only use bullet points or paragraphs.
    """
    if not PERPLEXITY_API_KEY:
        return "API key not available for summary generation. Please configure your PERPLEXITY_API_KEY."
    try:
        answer, citations = get_perplexity_answer(prompt)
        return convert_table_to_bullets(answer)
    except Exception as e:
        return f"Could not generate summary due to API error: {e}"

def is_aggregator_title(title):
    """More lenient aggregator detection"""
    if not title:
        return False
        
    blacklist_keywords = ['top', 'best', 'guide', 'list', 'review']
    title_lower = title.lower()
    
    # Only exclude if title clearly indicates it's an aggregator
    if any(title_lower.startswith(word + " ") for word in blacklist_keywords):
        return True
        
    # Count aggregator words - if too many, likely an aggregator
    aggregator_count = sum(1 for word in blacklist_keywords if word in title_lower)
    return aggregator_count >= 2

def extract_manufacturer_info(company_name):
    prompt = f"Find the official website and LinkedIn page for the company named '{company_name}'. Return both URLs if available. Prioritize the official website over LinkedIn. Format: Official Website: [URL], LinkedIn: [URL]"
    answer, _ = get_perplexity_answer(prompt)

    linkedin_url = None
    website_url = None
    website_candidates = []

    for line in answer.splitlines():
        url_matches = re.findall(r'https?://(?:www\.)?[^\s/$.?#].[^\s]*', line)
        for url in url_matches:
            if "linkedin.com" in url.lower():
                linkedin_url = url
            elif url.startswith("http") and "linkedin.com" not in url.lower():
                website_candidates.append(url)
    
    if website_candidates:
        website_url = website_candidates[0]

    return linkedin_url, website_url

def parse_search_results(answer):
    # Try JSON first
    try:
        data = json.loads(answer)
        if isinstance(data, list):
            return {'organic': data}
        if isinstance(data, dict) and 'organic' in data:
            return data
    except Exception:
        pass
    
    # Try to extract JSON array using regex
    json_match = re.search(r'\[\s*\{.*?\}\s*\]', answer, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if isinstance(data, list):
                return {'organic': data}
        except Exception:
            pass
    
    # Try Python literal
    try:
        data = ast.literal_eval(answer)
        if isinstance(data, list):
            return {'organic': data}
    except Exception:
        pass
    
    # Try to extract markdown-style links
    results = []
    for match in re.finditer(r'\[([^\]]+)\]\((https?://[^\)]+)\)', answer):
        title, url = match.groups()
        results.append({'title': title, 'link': url, 'snippet': ''})
    
    if results:
        return {'organic': results}
    
    # Last resort: extract any URLs with their context as title
    url_pattern = r'https?://[^\s]+'
    urls = re.findall(url_pattern, answer)
    for url in urls:
        # Find the text before the URL as potential title
        context = answer[:answer.find(url)].strip().split('\n')[-1] if url in answer else ""
        results.append({'title': context or url, 'link': url, 'snippet': ''})
    
    if results:
        return {'organic': results}
    
    return None

def convert_table_to_bullets(text):
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
        return text

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
    
    non_table = [line for line in lines if line not in table_lines]
    return "\n".join(non_table) + "\n" + "\n".join(bullets)

def parse_user_prompt_for_search(user_prompt):
    max_results = 5
    match = re.search(r'\b(\d+)\b', user_prompt)
    if match:
        num = int(match.group(1))
        if 1 <= num <= 20:
            max_results = num

    prompt_for_llm = f"""
    You are a search query generation expert. Based on the user's request below, create a JSON array of 3-5 diverse and effective Google search queries to find relevant manufacturers.
    The queries should be specific and varied to maximize the chances of finding unique, high-quality results.
    
    **CRITICAL: Focus on finding official company websites, NOT blogs, directories, or review sites.**
    
    User request: "{user_prompt}"
    
    Example for "Find me 10 bespoke furniture manufacturers in Italy":
    ["bespoke furniture manufacturer Italy official website", 
     "custom furniture company Italy contact", 
     "Italian furniture manufacturers directory",
     "luxury furniture makers Italy"]
    
    Respond ONLY with the JSON array, nothing else.
    """
    try:
        answer, _ = get_perplexity_answer(prompt_for_llm)
        search_terms = json.loads(answer)
        if isinstance(search_terms, list) and all(isinstance(term, str) for term in search_terms):
            return max_results, search_terms
    except (json.JSONDecodeError, TypeError):
        pass
    
    return max_results, [user_prompt.replace('"', '')]
    

def process_manufacturer_candidate(candidate):
    """Process a single manufacturer candidate in parallel"""
    company_title = candidate.get('title', candidate.get('name', ''))
    website_url = candidate.get('link', candidate.get('website_url', ''))
    
    if not website_url or not company_title:
        return None
    
    # Skip obvious aggregators and unwanted domains - but be more lenient
    hard_skip_domains = [
        'made-in-china.com', 'alibaba.com', 'thomasnet.com', 'quora.com',
        'reddit.com', 'facebook.com', 'homedepot.com', 'indiamart.com',
        'justdial.com', 'yelp.com', 'amazon.com', 'ebay.com', 'wikipedia.org'
    ]
    
    if any(domain in website_url.lower() for domain in hard_skip_domains):
        return None
    
    # More lenient title filtering - only exclude obvious aggregators
    if is_aggregator_title(company_title):
        return None
    
    # Get LinkedIn URL
    linkedin_url, _ = extract_manufacturer_info(company_title)
    
    # Scrape website content
    combined_content, all_emails, all_phones, all_addresses = scrape_manufacturer_website_parallel(website_url)
    if not combined_content:
        return None
    
    # Generate summary
    summary = generate_manufacturer_summary_from_content(company_title, combined_content)
    
    return {
        "name": company_title,
        "website_url": website_url,
        "linkedin_url": linkedin_url,
        "summary": summary,
        "emails": all_emails,
        "phones": all_phones,
        "addresses": all_addresses,
        "extracted_content": combined_content
    }

def perform_deep_dive_analysis(supplier):
    
    search_filters = [
        "Location & Logistics",
        "Minimum Order Quantity (MOQ)",
        "Lead Time & Delivery",
        "Certifications & Quality Standards", 
        "Past Project Experience & Clients",
        "Manufacturing Capabilities & Capacity",
        "Customization & Engineering Support",
        "Quality Control Process",
        "Export Experience & Shipping",
        "Company History & Reputation"
    ]
    
    all_collected_data = {}
    
    for i, filter_query in enumerate(search_filters):
        query = f"{supplier['name']} {filter_query} official information"
        
        # Use Perplexity for targeted search
        prompt = f"""
        Find specific, concrete information about: "{query}"
        Return ONLY factual information about this company's {filter_query.lower()}.
        Include numbers, dates, specific projects, certifications, or capabilities when available.
        If no specific information is found, state "No specific information found".
        """
        
        answer, _ = get_perplexity_answer(prompt)
        all_collected_data[filter_query] = answer
    
    # Format collected data
    collected_data_text = f"COMPANY: {supplier['name']}\nWEBSITE: {supplier['website_url']}\n\n"
    
    for category, info in all_collected_data.items():
        collected_data_text += f"=== {category.upper()} ===\n{info}\n\n"
    
    # Add previously extracted content
    if supplier.get('extracted_content'):
        collected_data_text += f"=== WEBSITE CONTENT EXTRACTED ===\n{supplier['extracted_content'][:5000]}\n\n"
    
    # Generate comprehensive analysis
    final_prompt = f"""
    You are a professional business analyst. Based on ALL the collected information below, provide a COMPREHENSIVE business analysis of {supplier['name']}.
    
    STRUCTURE YOUR ANALYSIS WITH THESE SECTIONS:
    
    1. **COMPANY OVERVIEW**
       - Business focus and specialization
       - Company size and history
       - Market position and reputation
    
    2. **PRODUCTS & SERVICES**
       - Core product offerings
       - Specializations and unique capabilities
       - Product quality and standards
    
    3. **MANUFACTURING CAPABILITIES**
       - Production capacity and scale
       - Factory information and locations
       - Technical capabilities and equipment
    
    4. **BUSINESS OPERATIONS**
       - Minimum order requirements
       - Lead times and delivery capabilities
       - Quality control processes
       - Certifications and compliance
    
    5. **CLIENT EXPERIENCE & PROJECTS**
       - Past projects and notable clients
       - Industry experience and specialties
       - Client support and services
    
    6. **CONTACT & LOGISTICS**
       - Contact information summary
       - Shipping and export experience
       - Geographic reach and capabilities
    
    7. **STRENGTHS & COMPETITIVE ADVANTAGES**
       - Key differentiators
       - Unique selling propositions
       - Market advantages
    
    8. **BUSINESS SUMMARY & RECOMMENDATION**
       - Overall assessment
       - Suitability for partnerships
       - Key considerations
    
    **IMPORTANT:**
    - Be specific and include numbers, dates, names when available
    - Use bullet points for clarity
    - Highlight both strengths and potential limitations
    - If information is not available, state "Information not available"
    - Do not invent information - only use what's provided
    
    COLLECTED INFORMATION:
    {collected_data_text}
    """
    
    try:
        analysis_result, _ = get_perplexity_answer(final_prompt)
        return analysis_result
    except Exception as e:
        return f"Deep analysis failed: {str(e)}"

def run_app():
    st.markdown('<h1 style="color:#BC8A7E;">Wood Couture AI Market Scout</h1>', unsafe_allow_html=True)
    st.markdown("I can help you search and analyze bespoke furniture related manufacturers worldwide.")

    # Initialize session state
    for key in ['suppliers_for_analysis', 'last_general_search_results', 'last_specific_company_summary', 
                'analysis_results', 'analyzed_suppliers', 'current_search_mode', 'search_in_progress']:
        if key not in st.session_state:
            if key == 'suppliers_for_analysis':
                st.session_state[key] = []
            elif key == 'last_general_search_results':
                st.session_state[key] = {}
            elif key == 'analysis_results':
                st.session_state[key] = {}
            elif key == 'analyzed_suppliers':
                st.session_state[key] = set()
            elif key == 'current_search_mode':
                st.session_state[key] = "General Search"
            elif key == 'search_in_progress':
                st.session_state[key] = False
            else:
                st.session_state[key] = None

    search_tab, analysis_tab = st.tabs(["🔍 Search", "📊 Analysis Results"])

    with search_tab:
        st.session_state['current_search_mode'] = st.selectbox(
            "Select Search Mode",
            ("General Search", "Specific Company Search"),
            key="search_mode_selectbox",
        )

        # General Search
        if st.session_state['current_search_mode'] == "General Search":
            st.header("General Manufacturer Search")
            user_prompt = st.text_area(
                "Describe the manufacturers you want to find:",
                placeholder="e.g., 'Find me 3 bespoke furniture manufacturers in China specializing in office furniture'",
                height=100,
                key="general_search_prompt"
            )
            
            col1, col2 = st.columns([1, 1])
            with col1:
                search_clicked = st.button("Perform General Search", 
                                         key="perform_general_search_button")
            
            if search_clicked:
                if not PERPLEXITY_API_KEY:
                    st.error("❌ Perplexity API key not configured.")
                elif not user_prompt.strip():
                    st.error("❌ Please enter a search description.")
                else:
                    st.session_state['search_in_progress'] = True
                    st.session_state['last_general_search_results'] = {}
                    
                    with st.spinner("🔍 Analyzing your request and preparing search queries..."):
                        max_results, search_terms = parse_user_prompt_for_search(user_prompt)
                        st.info(f"Searching for up to {max_results} companies using {len(search_terms)} search strategies.")

                    all_candidates = []
                    seen_urls = set()

                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Collect candidates from all search terms
                    total_searches = len(search_terms)
                    for search_idx, term in enumerate(search_terms):
                        status_text.text(f"🔎 Search {search_idx + 1}/{total_searches}: '{term}'")
                        prompt = (
                            f"Search for the query: '{term}' and return ONLY official company websites. "
                            "Exclude blogs, directories, review sites, marketplaces, and aggregators. "
                            "Respond ONLY with a valid JSON array of objects, each with 'title', 'link', and 'snippet' fields. "
                            "Prioritize websites that look like official company websites. "
                            f"Return at least {max_results} results if possible. "
                            "Example: [{\"title\": \"Company A\", \"link\": \"https://companya.com\", \"snippet\": \"Official website of Company A...\"}]"
                        )
                        answer, _ = get_perplexity_answer(prompt)
                        search_results = parse_search_results(answer)

                        if search_results and 'organic' in search_results:
                            for result in search_results['organic']:
                                if len(all_candidates) >= max_results + 2:
                                    break
                                website_url = result.get('link', '').strip('". ')
                                if website_url and website_url not in seen_urls:
                                    seen_urls.add(website_url)
                                    all_candidates.append(result)
                        
                        progress_bar.progress((search_idx + 1) / total_searches)

                    # Process candidates in parallel
                    if all_candidates:
                        status_text.text(f"Processing {len(all_candidates)} candidate companies in parallel...")
                        st.info(f"Found {len(all_candidates)} candidate companies. Processing in parallel...")
                        
                        processing_bar = st.progress(0)
                        results_dict = {}
                        successful_count = 0
                        
                        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                            future_to_candidate = {
                                executor.submit(process_manufacturer_candidate, candidate): candidate 
                                for candidate in all_candidates[:max_results * 2]
                            }
                            
                            completed = 0
                            total = len(future_to_candidate)
                            
                            for future in as_completed(future_to_candidate):
                                try:
                                    result = future.result()
                                    if result and len(results_dict) < max_results:
                                        results_dict[result['name']] = result
                                        successful_count += 1
                                        st.success(f"✅ Processed: {result['name']}")
                                except Exception as e:
                                    pass
                                
                                completed += 1
                                if total > 0:
                                    processing_bar.progress(completed / total)
                        
                        st.session_state['last_general_search_results'] = results_dict
                        st.session_state['search_in_progress'] = False
                        
                        if results_dict:
                            st.balloons()
                            st.success(f"🎉 Search complete! Found {len(results_dict)} companies ready for analysis.")
                        else:
                            st.warning("⚠️ Found candidate companies but couldn't process them successfully. Try adjusting your search terms to be more specific about official company websites.")
                    else:
                        st.session_state['search_in_progress'] = False
                        st.error("❌ No manufacturers found. Please try a different prompt or be more specific about the type of manufacturers you're looking for.")
            st.markdown("""
            <style>
            div.streamlit-expanderHeader p {
                font-size: 22px !important;
                font-weight: 700 !important;
                color: #111111 !important;
            }
            </style>
            """, unsafe_allow_html=True)
            # Display results
            if (st.session_state['last_general_search_results'] and 
                not st.session_state.get('search_in_progress', False)):
                st.subheader("Found Manufacturers:")

                for idx, (company_name, details) in enumerate(st.session_state['last_general_search_results'].items(), 1):
                    with st.expander(f"{idx}. {details['name']}", expanded=True):
                        st.markdown(
                            f"<h3 style='font-size:22px; color:#111111; font-weight:700;'>{details['name']}</h3>",
                            unsafe_allow_html=True
                        )
                        st.write(f"**Website:** {details['website_url']}")
                        if details['linkedin_url']:
                            st.write(f"**💼 LinkedIn:** {details['linkedin_url']}")
                        
                        if details.get('emails'):
                            st.write(f"**Emails:** {', '.join(details['emails'][:3])}")
                        if details.get('phones'):
                            st.write(f"**Phones:** {', '.join(details['phones'][:3])}")
                        
                        st.markdown("**Summary:**")
                        st.markdown(details['summary'])
                        
                        col1, col2= st.columns([1, 1])
                        with col1:
                            if st.button(f"Add to Analysis", key=f"add_{details['name']}"):
                                if not any(s['website_url'] == details['website_url'] for s in st.session_state['suppliers_for_analysis']):
                                    st.session_state['suppliers_for_analysis'].append(details)
                                    st.success(f"✅ {details['name']} added to analysis box!")
                                else:
                                    st.info(f"ℹ️ {details['name']} is already in the analysis box.")
                        
                        with col2:
                            pdf_filename = f"{details['name'].replace(' ', '_')}_Summary.pdf"
                            pdf_content = generate_pdf_from_text(details['summary'], details['name'], details['website_url'], pdf_filename)
                            st.download_button(
                                label="📄 Download PDF",
                                data=pdf_content,
                                file_name=pdf_filename,
                                mime="application/pdf",
                                key=f"download_{details['name']}"
                            )
                        


        # Specific Company Search
        if st.session_state['current_search_mode'] == "Specific Company Search":
            st.header("Specific Company Search")
            specific_company_input = st.text_input(
                "Enter the exact name of the company/manufacturer:",
                key="specific_company_name_input",
                placeholder="e.g., 'Steelcase Inc.' or 'Herman Miller'"
            )

            if st.button("🔍 Search Specific Company", key="search_specific_company_btn"):
                if specific_company_input:
                    with st.spinner(f"🔍 Searching for {specific_company_input}..."):
                        linkedin_url, website_url = extract_manufacturer_info(specific_company_input)
                    
                    if not website_url:
                        st.error("❌ Could not find a valid website for the specified company.")
                    else:
                        st.write(f"**🌐 Website:** {website_url}")
                        if linkedin_url:
                            st.write(f"**💼 LinkedIn:** {linkedin_url}")
                            
                        with st.spinner("🔄 Scraping website and generating summary..."):
                            combined_content, all_emails, all_phones, all_addresses = scrape_manufacturer_website_parallel(website_url)
                            if not combined_content:
                                st.error("❌ Failed to extract meaningful content from the website.")
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
                                st.success("✅ Specific company analysis complete!")

            if st.session_state['last_specific_company_summary']:
                details = st.session_state['last_specific_company_summary']
                st.subheader(f"🏢 Manufacturer: {details['name']}")
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write(f"**🌐 Website:** {details['website_url']}")
                    if details['linkedin_url']:
                        st.write(f"**💼 LinkedIn:** {details['linkedin_url']}")
                    
                    if details.get('emails'):
                        st.write(f"**📧 Emails:** {', '.join(details['emails'][:3])}")
                    if details.get('phones'):
                        st.write(f"**📞 Phones:** {', '.join(details['phones'][:3])}")
                
                with col2:
                    if st.button(f"Add to Analysis Box", key=f"add_specific_{details['name']}"):
                        if not any(s['website_url'] == details['website_url'] for s in st.session_state['suppliers_for_analysis']):
                            st.session_state['suppliers_for_analysis'].append(details)
                            st.success(f"✅ {details['name']} added to analysis box!")
                        else:
                            st.info(f"ℹ️ {details['name']} is already in the Side Bar.")
                    
                    pdf_filename = f"{details['name'].replace(' ', '_')}_Summary.pdf"
                    pdf_content = generate_pdf_from_text(details['summary'], details['name'], details['website_url'], pdf_filename)
                    st.download_button(
                        label="📄 Download PDF",
                        data=pdf_content,
                        file_name=pdf_filename,
                        mime="application/pdf",
                        key=f"download_specific_{details['name']}"
                    )
                
                st.markdown("**📋 Summary:**")
                st.markdown(details['summary'])

    with analysis_tab:
        st.header("Supplier Analysis")
        
        if not st.session_state['suppliers_for_analysis']:
            st.info("No suppliers to analyze. Please add suppliers from the Search tab first.")
        else:
            # Summary statistics
            total_suppliers = len(st.session_state['suppliers_for_analysis'])
            analyzed_count = len(st.session_state['analyzed_suppliers'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Suppliers", total_suppliers)
            with col2:
                st.metric("Analyzed", analyzed_count)
            with col3:
                st.metric("Pending", total_suppliers - analyzed_count)
            
            st.subheader("📋 Suppliers Available for Analysis")
            
            # Create a better display for suppliers
            for i, supplier in enumerate(st.session_state['suppliers_for_analysis']):
                with st.expander(f"{i+1}. {supplier['name']}", expanded=False):
                    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                    
                    with col1:
                        st.write(f"**Website:** {supplier['website_url']}")
                        if supplier.get('linkedin_url'):
                            st.write(f"**LinkedIn:** {supplier['linkedin_url']}")
                    
                    with col2:
                        status = "✅ Analyzed" if supplier['name'] in st.session_state['analyzed_suppliers'] else "⏳ Pending"
                        st.write(f"**Status:** {status}")
                    
                    with col3:
                        if st.button("Analyze", key=f"analyze_btn_{i}", use_container_width=True):
                           with st.spinner(f"Analysing"):
                                analysis_result = perform_deep_dive_analysis(supplier)
                                st.session_state['analysis_results'][supplier['name']] = analysis_result
                                st.session_state['analyzed_suppliers'].add(supplier['name'])
                                st.success(f"✅ Analysis complete for {supplier['name']}!")
                                st.rerun()
                    
                    with col4:
                        if st.button("Remove", key=f"remove_btn_{i}", use_container_width=True):
                            if supplier['name'] in st.session_state['analyzed_suppliers']:
                                st.session_state['analyzed_suppliers'].remove(supplier['name'])
                            if supplier['name'] in st.session_state['analysis_results']:
                                del st.session_state['analysis_results'][supplier['name']]
                            st.session_state['suppliers_for_analysis'].pop(i)
                            st.rerun()

                    # Show analysis results if available
                    if supplier['name'] in st.session_state['analyzed_suppliers']:
                        st.markdown("---")
                        st.subheader(f"Analysis Results for {supplier['name']}")
                        analysis_result = st.session_state['analysis_results'][supplier['name']]
                        st.markdown(analysis_result)
                        
                        # Download button for analysis results
                        pdf_filename = f"{supplier['name'].replace(' ', '_')}_Deep_Analysis.pdf"
                        pdf_content = generate_pdf_from_text(analysis_result, supplier['name'], supplier['website_url'], pdf_filename)
                        st.download_button(
                            label="📄 Download Deep Analysis PDF",
                            data=pdf_content,
                            file_name=pdf_filename,
                            mime="application/pdf",
                            key=f"download_deep_{supplier['name']}",
                            use_container_width=True
                        )
            print(f"Performing deep dive analysis for {supplier['name']}...")
            
            st.markdown("---")
            
            # Bulk actions
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Analyze All Suppliers", key="analyze_all_btn", use_container_width=True):
                    for i, supplier in enumerate(st.session_state['suppliers_for_analysis']):
                        if supplier['name'] not in st.session_state['analyzed_suppliers']:
                            with st.spinner(f"Analyzing {supplier['name']} ({i+1}/{len(st.session_state['suppliers_for_analysis'])})..."):
                                analysis_result = perform_deep_dive_analysis(supplier)
                                st.session_state['analysis_results'][supplier['name']] = analysis_result
                                st.session_state['analyzed_suppliers'].add(supplier['name'])
                    st.success("🎉 All suppliers analyzed!")
                    st.rerun()
            
            with col2:
                if st.button("🔄 Clear All Results", key="clear_all_results", use_container_width=True):
                    st.session_state['analyzed_suppliers'] = set()
                    st.session_state['analysis_results'] = {}
                    st.success("✅ All analysis results cleared!")
                    st.rerun()
            
            with col3:
                if st.button("🗑️ Clear All Suppliers", key="clear_all_suppliers", use_container_width=True):
                    st.session_state['suppliers_for_analysis'] = []
                    st.session_state['analyzed_suppliers'] = set()
                    st.session_state['analysis_results'] = {}
                    st.success("✅ All suppliers cleared!")
                    st.rerun()

    # Enhanced Sidebar
    st.sidebar.header("Suppliers for Analysis")
    
    if st.session_state['suppliers_for_analysis']:
        analyzed_count = len(st.session_state['analyzed_suppliers'])
        total_count = len(st.session_state['suppliers_for_analysis'])
        st.sidebar.info(f"Total suppliers: {total_count} | Analyzed: {analyzed_count}")
        #st.sidebar.metric("Suppliers in Box", total_count)
        #st.sidebar.metric("Analyzed", analyzed_count)
        
        progress_value = analyzed_count / total_count if total_count > 0 else 0
        st.sidebar.progress(progress_value)
        st.sidebar.caption(f"Analysis Progress: {analyzed_count}/{total_count}")
        
        st.sidebar.markdown("### Supplier List:")
        for supplier in st.session_state['suppliers_for_analysis']:
            status = "✅" if supplier['name'] in st.session_state['analyzed_suppliers'] else "⏳"
            st.sidebar.write(f"{status} {supplier['name']}")
        
        st.sidebar.markdown("---")
        
        # Quick actions in sidebar
        if st.sidebar.button("Analyze All", key="sidebar_analyze_all", use_container_width=True):
            st.session_state['perform_analysis'] = True
            st.session_state['analyze_all'] = True
            st.rerun()
            
    else:
        st.sidebar.info("No suppliers added for analysis yet.")


if __name__ == "__main__":
    run_app()