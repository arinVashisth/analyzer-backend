from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import re
import spacy

##### LLM #####
from difflib import SequenceMatcher

##### NLP #####
nlp = spacy.load("en_core_web_sm")

ALL_KNOWN_SKILLS = {
    # Programming Languages
    "python", "java", "c++", "c", "c#", "javascript", "typescript", "go", "rust", "ruby", "swift", "kotlin", "r", "scala", "perl", "dart",

    # Frontend
    "html", "css", "js", "react", "next.js", "vue", "angular", "svelte", "tailwindcss", "bootstrap", "sass", "redux", "jquery",

    # Backend
    "node.js", "express", "flask", "fastapi", "django", "spring", "rails", "nestjs", "graphql", "rest", "api", "grpc",

    # Databases
    "sql", "sqlite", "postgresql", "mysql", "mongodb", "redis", "cassandra", "elasticsearch", "dynamodb", "firebase",

    # DevOps / Cloud / Infra
    "docker", "kubernetes", "aws", "gcp", "azure", "firebase", "heroku", "netlify", "vercel", "terraform", "ansible", "jenkins", "ci/cd", "bash", "linux", "shell", "nginx", "apache",

    # ML / AI / Data
    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "sklearn", "tensorflow", "pytorch", "keras", "xgboost", "lightgbm", "catboost", "huggingface", "transformers", "openai", "langchain", "mlflow",

    # Data Engineering / Analytics
    "airflow", "dbt", "spark", "hadoop", "kafka", "dask", "superset", "tableau", "powerbi", "excel",

    # Other Tools & Concepts
    "git", "github", "gitlab", "bitbucket", "jira", "confluence", "notion", "postman", "swagger", "unit testing", "tdd", "bdd", "oop", "design patterns", "data structures", "algorithms", "nlp", "mle", "llm", "cv", "mleops", "rag", "prompt engineering", "weaviate", "pinecone", "qdrant", "vector db",

    # Security / Networking (basic entries)
    "oauth", "jwt", "ssl", "tls", "cors", "firewall", "tcp/ip", "dns",

    # Mobile & Cross-Platform
    "react native", "flutter", "android", "ios", "expo",

    # Testing / QA
    "pytest", "unittest", "selenium", "cypress", "jest", "mocha", "chai"
}

SKILL_ALIASES = {
    "scikit-learn": "sklearn",
    "js": "javascript",
    "tf": "tensorflow",
    "ts": "typescript"
}

ACTION_VERBS = [
    'developed', 'built', 'created', 'designed', 'engineered', 'implemented',
    'led', 'managed', 'deployed', 'researched', 'integrated', 'explored',
    'worked', 'focused', 'collaborated', 'automated', 'maintained'
]

def normalize_skills(skills):
    return {SKILL_ALIASES.get(skill, skill) for skill in skills}

def fuzzy_skill_match(resume_skills, jd_skills, threshold=0.85):
    matched = set()
    for jd in jd_skills:
        for rs in resume_skills:
            if SequenceMatcher(None, jd, rs).ratio() >= threshold:
                matched.add(jd)
                break
    missing = jd_skills - matched
    return sorted(matched), sorted(missing)


########################################## Spacy ##############################################

def extract_skills_with_ner(text):
    doc = nlp(text)
    skills = set()

    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART","SKILLS","SKILL"]:
            skills.add(ent.text.lower())

    return list(skills)

########################################### Functions ##############################################

def extract_name(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    
    # Step 1: Locate line with phone number
    phone_pattern = r'\+?\d[\d\s\-\(\)]{8,}'
    phone_index = -1
    for i, line in enumerate(lines):
        if re.search(phone_pattern, line):
            phone_index = i
            break
    
    if phone_index == -1:
        return ""

    # Step 2: Look up to 3 lines before phone number
    candidate_lines = lines[max(0, phone_index - 3): phone_index]

    # Combine all candidate lines into one string
    raw = " ".join(candidate_lines)

    # Step 3: Remove unwanted characters, numbers
    raw = re.sub(r'[^A-Za-z\s]', '', raw)

    # Step 4: If the name has a pattern like "A N J O L I", fix it
    if len(raw.replace(" ", "")) <= 20:
        words = raw.split()
        if all(len(w) == 1 for w in words):  # spaced letters
            name = "".join(words).capitalize()
        else:
            name = " ".join([w.capitalize() for w in words])
    else:
        name = raw.strip().title()
    name_list = name.split()
    if len(name_list)>1:
        name = name_list[0]+" "+name_list[1]
    else:
        name = name_list[0]
    return name

def extract_mobile_number(text):
    # Match most common Indian phone formats
    phone_pattern = r'(\+91[\-\s]?)?(\(?\d{3,5}\)?[\s\-]?\d{3,5}[\s\-]?\d{3,5})'
    matches = re.findall(phone_pattern, text)

    if matches:
        # Return cleaned first match
        raw_number = ''.join(matches[0])
        # Remove spaces, dashes, parentheses
        number = re.sub(r'[^\d]', '', raw_number)
        # Optional: Keep only 10-digit mobile numbers
        if len(number) >= 10:
            return number[-10:]
    return ""

def extract_email(text):
    # Basic but effective regex for most email patterns
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    matches = re.findall(email_pattern, text)

    if matches:
        return matches[0]  # Return the first valid email
    return ""

def extract_links(text):
    # Match both full and partial URLs (e.g., github.com/... or linkedin.com/...)
    pattern = r'(https?://[^\s|]+|www\.[^\s|]+|github\.com/[^\s|]+|linkedin\.com/[^\s|]+|[\w.-]+\.dev[^\s|]*|[\w.-]+\.me[^\s|]*)'
    matches = re.findall(pattern, text)

    links = {
        "linkedin": "",
        "github": "",
        "portfolio": "",
        "others": []
    }

    for url in matches:
        url = url.strip()

        # Ensure it's treated as a valid URL
        if not url.startswith("http"):
            url = "https://" + url

        if "linkedin.com" in url:
            links["linkedin"] = url
        elif "github.com" in url:
            links["github"] = url
        elif any(domain in url for domain in ["portfolio", ".dev", ".me", ".design"]):
            links["portfolio"] = url
        else:
            links["others"].append(url)

    return links

def extract_education(text):
    education_entries = []

    # 1. Locate the EDUCATION section only
    match = re.search(r"(education[:\n\r]+)([\s\S]{0,350})", text, re.IGNORECASE)
    if not match:
        return []

    edu_section = match.group(2)

    # 2. Stop if another section starts (like Experience, Projects, Skills)
    stop_keywords = ['experience', 'project', 'internship', 'skill', 'work', 'achievement','internships']
    for stop_word in stop_keywords:
        stop_idx = edu_section.lower().find(stop_word)
        if stop_idx != -1:
            edu_section = edu_section[:stop_idx]
            break

    # 3. Keywords for filtering valid education lines
    degree_keywords = [
        'b.tech', 'btech', 'be', 'bs', 'b.sc', 'bsc', 'bachelor',
        'm.tech', 'mtech', 'me', 'ms', 'm.sc', 'msc', 'master',
        'phd', 'mba', 'b.com', 'm.com'
    ]
    degree_regex = re.compile(r'(' + '|'.join(degree_keywords) + r')', re.IGNORECASE)
    year_regex = re.compile(r'(20\d{2}|19\d{2}|present)', re.IGNORECASE)

    # Optional: common verbs that imply descriptions
    noisy_verbs = ['built', 'developed', 'focused', 'explored', 'worked', 'engineered', 'created']

    lines = edu_section.strip().split('\n')

    for line in lines:
        line_clean = line.strip().lower()
        if any(verb in line_clean for verb in noisy_verbs):
            continue  # Skip non-education lines
        if degree_regex.search(line_clean) or year_regex.search(line_clean):
            education_entries.append(line.strip())

    return education_entries[:5]  # Limit to top 5 entries

def extract_experience(text):
    experience_entries = []

    # 1. Locate Experience/Internship section
    match = re.search(r"(experience|work experience|internship|professional experience|internships)[:\n\r]+([\s\S]{0,1000})", text, re.IGNORECASE)
    if not match:
        return []

    exp_section = match.group(2)

    # 2. Stop when hitting another section
    stop_keywords = ['education', 'project', 'skill', 'activity', 'achievement', 'certification']
    for stop_word in stop_keywords:
        stop_idx = exp_section.lower().find(stop_word)
        if stop_idx != -1:
            exp_section = exp_section[:stop_idx]
            break

    lines = exp_section.strip().split('\n')

    # 3. Filter lines that likely contain experience content
    for line in lines:
        line_clean = line.strip().lower()
        if any(verb in line_clean for verb in ACTION_VERBS):
            experience_entries.append(line.strip())
        elif re.search(r'\d{4}', line):  # likely year-based experience line
            experience_entries.append(line.strip())
        elif line.strip().startswith(('•', '-', '*')):
            experience_entries.append(line.strip())

    return experience_entries[:7]  # return top 7 entries max

def extract_projects(text):
    text = text.lower()
    lines = text.splitlines()

    structured_projects = []
    i = 0

    email_pattern = re.compile(r'\S+@\S+')
    phone_pattern = re.compile(r'\b(\+91[-\s]?)?\d{10}\b')
    link_pattern = re.compile(r'(https?://|www\.|linkedin\.com|github\.com|\.com)')

    while i < len(lines):
        line = lines[i].strip()

        # ⛔ Skip known contact lines
        if email_pattern.search(line) or phone_pattern.search(line) or link_pattern.search(line):
            i += 1
            continue

        # ✅ Check if it's a project title/tech line
        if ('|' in line or re.search(r'\b(flask|react|node|vue|api|github|project|ml|fastapi|postgresql|python|html|css|javascript)\b', line)) and len(line) > 10:
            title_line = line
            description = ""

            # Check for bullet description in next line
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line.startswith("•") or re.search(r'(built|developed|implemented|created|deployed)', next_line):
                    description = next_line.lstrip('•').strip().capitalize()
                    i += 1

            # Try to split title and tech stack
            parts = re.split(r'\s*\|\s*', title_line)
            if len(parts) >= 2:
                title = parts[0].strip().title()
                tech_stack = parts[1].strip()
            else:
                title = title_line.strip().title()
                tech_stack = ""

            structured_projects.append({
                "title": title,
                "tech_stack": tech_stack,
                "description": description
            })

        i += 1

    return structured_projects

def filter_valid_projects(projects):
    filtered = []
    for proj in projects:
        title = proj['title'].lower()
        tech = proj['tech_stack'].lower()

        # Skip lines that are emails, phone numbers, or just skills
        if re.search(r'\b(gmail|linkedin|github|\.com|^\d{10,})\b', title):
            continue
        if 'languages:' in title or 'frameworks:' in title or 'databases:' in title:
            continue
        if len(title.split()) < 2 and not tech:  # too short and no tech = not valid
            continue
        if not re.search(r'(project|app|system|platform|tool|portal)', title):
            if len(tech.strip()) < 5:
                continue

        filtered.append(proj)

    return filtered

#################################################  TEXT PART  #######################################################

def build_clean_resume_text(resume_text, skills):
    # Extract each section
    experience = extract_experience(resume_text)
    projects = filter_valid_projects(extract_projects(resume_text))
    

    # Create a clean string for each section
    experience_text = ""
    for line in experience[:3]:  # Limit to top 5
        line = line.strip()
        if not line.endswith((".", "!", "?")):
            line += "."
        experience_text += f"- {line}\n"

    projects_text = ""
    for p in projects[:2]:  # Limit to top 2
        projects_text += f"\n• {p['title']} | {p['tech_stack']}\n  {p['description']}"

    skills_text = ', '.join(skills) if skills else "N/A"

    final_text = (
        f"Experience:\n{experience_text}\n"
        f"Projects:{projects_text}\n\n"
        f"Skills: {skills_text}"
    )

    return final_text.strip()


def build_clean_jd_text(jd_text):
    # Remove emojis
    jd_text = re.sub(r'[^\x00-\x7F]+', '', jd_text)

    # Keep only relevant sections
    keep_blocks = [
        "job title",
        "responsibilities",
        "required skills",
        "good to have",
        "nice to have"
    ]

    lines = jd_text.lower().splitlines()
    cleaned_lines = []
    capture = False

    for line in lines:
        line = line.strip()
        if any(key in line for key in keep_blocks):
            capture = True
            cleaned_lines.append(line.capitalize())
        elif re.match(r'^(eligibility|what you|apply|location|duration)', line):
            capture = False
        elif capture and line:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()

#################################################  SKILLS PART  #######################################################
def extract_filtered_ner_skills(text, known_skills):
    """
    Run NER and return only relevant skill keywords by filtering against a known skill list.
    """
    doc = nlp(text)
    ner_skills = set()

    for ent in doc.ents:
        word = ent.text.lower().strip()
        if word in known_skills:
            ner_skills.add(word)
    return ner_skills

# Extract candidate technical skills from JD text
def extract_skills_from_jd(jd_text, known_skills):
    jd_text = jd_text.lower()
    extracted_skills = set()

    # Section headers that usually contain skills
    skill_sections = ["required skills", "good to have", "responsibilities", "what you'll do", "what you'll do"]

    for section in skill_sections:
        # Match the section and up to 6 following bullet points or 1000 chars
        pattern = rf"{section}[:\n\r\s-]*([\s\S]{{0,1000}}?)(?=\n\s*###|$)"
        match = re.search(pattern, jd_text, re.IGNORECASE)

        if match:
            block = match.group(1)
            block = block.translate(str.maketrans('', '', string.punctuation))
            words = set(block.strip().split())

            # Match only known skills
            for word in words:
                clean = word.strip().lower()
                if clean in known_skills:
                    extracted_skills.add(clean)

    return extracted_skills

# Tokenize resume and filter against valid skills
def clean_and_tokenize(text, valid_skills):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    return set(word for word in words if word in valid_skills)

# Extract structured skills section if present
def extract_skills_section(resume_text, valid_skills):
    text = resume_text.lower()
    
    # Match "skills" or "technical skills" block and grab up to 1000 chars after it
    pattern = r"(technical skills|skills)[\s:\-]*([\s\S]{0,1000})"
    match = re.search(pattern, text, re.IGNORECASE)

    if not match:
        return []

    block = match.group(2)
    
    # Remove punctuation
    block = block.translate(str.maketrans('', '', string.punctuation))

    # Extract words that are in the known skill list
    words = set(block.strip().split())
    extracted_skills = set()

    for word in words:
        clean = word.strip().lower()
        if clean in valid_skills:
            extracted_skills.add(clean)

    return list(extracted_skills)

# 🧠 Core matching function
def match_resume_with_jd(resume_text, jd_text):
    valid_skills = extract_skills_from_jd(jd_text,ALL_KNOWN_SKILLS)
    valid_skills.update(extract_filtered_ner_skills(jd_text,ALL_KNOWN_SKILLS))
    resume_skills = clean_and_tokenize(resume_text, ALL_KNOWN_SKILLS)
    resume_skills.update(extract_skills_section(resume_text, ALL_KNOWN_SKILLS))
    resume_skills.update(extract_filtered_ner_skills(resume_text,ALL_KNOWN_SKILLS))
    jd_skills = valid_skills
    resume_skills = normalize_skills(resume_skills)
    jd_skills = normalize_skills(jd_skills)
    clean_resume_text = build_clean_resume_text(resume_text, resume_skills)
    new_jd_text = build_clean_jd_text(jd_text)
    
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([clean_resume_text, new_jd_text])
    semantic_score = cosine_similarity(vectors[0], vectors[1])[0][0]

    matched_skills, missing_skills = fuzzy_skill_match(resume_skills, jd_skills) # LLM APPLIED
    skill_score = len(matched_skills) / len(jd_skills) if jd_skills else 0
    BOOST = 0.15
    semantic_score += BOOST
    semantic_score = max(0.0, min(semantic_score, 1.0))
    skill_score = max(0.0, min(skill_score, 1.0))
    if semantic_score < 0.5:
        hybrid_score = (0.4 * skill_score + 0.6 * semantic_score)
    else:
        hybrid_score = (0.3 * skill_score + 0.7 * semantic_score)
    hybrid_score = float(round(hybrid_score * 100, 2))


    # EXTRA THINGS:
    name = extract_name(resume_text)
    email = extract_email(resume_text)
    links = extract_links(resume_text)

    return hybrid_score, matched_skills, missing_skills, name, email, links


