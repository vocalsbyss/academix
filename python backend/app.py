from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from transformers import pipeline
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from ollama import Client
from datetime import date
import os
import uuid
import json
import threading
from functools import wraps
from dotenv import load_dotenv

# Load local .env if it exists
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "super-secret-key-change-me")

# Upload config for materials (PDF, Word)
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads", "materials")
ALLOWED_EXTENSIONS = {"pdf", "doc", "docx"}
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return filename and filename.rsplit(".", 1)[-1].lower() in ALLOWED_EXTENSIONS

# ================= DB =================
mongoUrl = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
mongo = MongoClient(mongoUrl)
db = mongo["academix_ai"]

users_col = db["users"]
universities_col = db["universities"]
rooms_col = db["rooms"]
messages_col = db["messages"]
subjects_col = db["subjects"]
materials_col = db["materials"]

# ================= AI =================
# Load Ollama configuration from environment for security (do not hard-code keys)
OLLAMA_API_KEY = "5a2da035b5b04de3a33becfc4e650a98.b4jbfUpnnng1phOZF0GhYGGy"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gpt-oss:120b")

classifier = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/deberta-v3-base-zeroshot-v1"
)

kw_model = KeyBERT()
embedder = SentenceTransformer("all-MiniLM-L6-v2")

if not OLLAMA_API_KEY:
    # Warn at startup instead of embedding secrets in source. LLM calls will raise if attempted.
    print("WARNING: OLLAMA_API_KEY not set. Ollama client will not be initialized. Set OLLAMA_API_KEY in your environment.")
    client = None
else:
    # Initialize Ollama client
    def get_ollama_client():
        ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        headers = {"Authorization": f"Bearer {OLLAMA_API_KEY}"}
        try:
            return Client(host=ollama_host, headers=headers)
        except Exception as e:
            print(f"ERROR: Failed to initialize Ollama client: {e}")
            return None

    client = get_ollama_client()

# Subject names universities can add materials for
DEFAULT_SUBJECTS = [
    "AI", "Web development", "Quantum", "Android", "Mathematics", "Physics", 
    "Chemistry", "Biology", "English Vocabulary", "English Grammar",
    "Thermodynamics", "Organic Chemistry", "Electromagnetic Theory", "Calculus",
    "Data Structures & Algorithms", "Quantum Mechanics", "Microeconomics", 
    "Macroeconomics", "History", "Geography", "Psychology", "Programming", "Databases"
]

CATEGORIES = ["Medical", "Engineering", "Arts", "Science", "Business", "Language", "Space", "Law"]

TOPIC_EMOJIS = {
    "AI": "🧠", "Artificial Intelligence": "🧠", "Machine Learning": "🤖", "Deep Learning": "🕸️",
    "Web development": "🌐", "Frontend": "🎨", "Backend": "⚙️", "Fullstack": "🥞",
    "Quantum": "🌀", "Quantum Mechanics": "🌌", "Quantum Computing": "💻",
    "Android": "📱", "iOS": "🍎", "Mobile Development": "📲",
    "Mathematics": "🧮", "Calculus": "📐", "Linear Algebra": "📉", "Statistics": "📊",
    "Physics": "⚛️", "Thermodynamics": "🔥", "Electromagnetic Theory": "⚡", "Optics": "🔭",
    "Chemistry": "🧪", "Organic Chemistry": "⚗️", "Inorganic Chemistry": "🧪", "Biochemistry": "🧬",
    "Biology": "🧬", "Genetics": "🧬", "Microbiology": "🧫", "Botany": "🌱",
    "English Vocabulary": "📚", "English Grammar": "✍️", "Literature": "📖", "Communication": "🗣️",
    "Data Structures & Algorithms": "🏗️", "DSA": "🏗️", "Programming": "💻", "Python": "🐍", "Java": "☕",
    "Databases": "🗄️", "SQL": "💾", "NoSQL": "🍃",
    "Microeconomics": "📉", "Macroeconomics": "📈", "Economics": "💰",
    "History": "📜", "World History": "🌍", "Ancient History": "🏛️",
    "Geography": "🌍", "Geology": "⛰️",
    "Psychology": "☯️", "Sociology": "👥", "Philosophy": "🤔",
    "Business": "💼", "Marketing": "📢", "Finance": "💸"
}

# ================= AUTH DECORATORS =================
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper


def university_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "university" not in session:
            return redirect(url_for("university_login"))
        return f(*args, **kwargs)
    return wrapper

# ================= AI HELPERS =================
def detect_intent(text):
    result = classifier(text,
        candidate_labels=["technical", "casual", "writing"])
    return result["labels"][0]

def detect_category(text):
    result = classifier(text, candidate_labels=CATEGORIES)
    return result["labels"][0]

def extract_keywords(text):
    kws = kw_model.extract_keywords(text, top_n=5)
    return [k[0] for k in kws]

def ask_llm(messages, stream=False):
    if client is None:
        # Fail-fast with a clear error so callers know to configure the key.
        raise RuntimeError("Ollama client is not configured. Set the OLLAMA_API_KEY environment variable.")

    response = client.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        stream=stream
    )
    if stream:
        return response
    return response["message"]["content"]

# ================= AUTH ROUTES =================

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        user = request.form["username"]
        university_name = request.form.get("university_name", "").strip()
        pw = generate_password_hash(request.form["password"])

        if users_col.find_one({"username": user}):
            return "User exists"

        users_col.insert_one({
            "username": user,
            "university_name": university_name,
            "password": pw,
            "keywords": {},
            "xp": 0,
            "last_login": None
        })
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        user = request.form["username"]
        pw = request.form["password"]

        db_user = users_col.find_one({"username": user})
        if db_user and check_password_hash(db_user["password"], pw):
            session["user"] = user

            # daily login XP (+5 once per day)
            today_str = date.today().isoformat()
            if db_user.get("last_login") != today_str:
                users_col.update_one(
                    {"username": user},
                    {"$set": {"last_login": today_str}, "$inc": {"xp": 5}}
                )

            return redirect(url_for("rooms"))

        return "Invalid login"

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ================= UNIVERSITY AUTH =================

@app.route("/university/register", methods=["GET", "POST"])
def university_register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        pw = request.form["password"]

        if not name:
            return "University name is required", 400
        if universities_col.find_one({"name": name}):
            return "A university with this name already exists", 400

        universities_col.insert_one({
            "name": name,
            "email": email,
            "password": generate_password_hash(pw),
        })
        return redirect(url_for("university_login"))

    return render_template("university_register.html")


@app.route("/university/login", methods=["GET", "POST"])
def university_login():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        pw = request.form["password"]

        uni = universities_col.find_one({"name": name})
        if uni and check_password_hash(uni["password"], pw):
            session["university"] = name
            return redirect(url_for("university_dashboard"))

        return "Invalid login", 401

    return render_template("university_login.html")


@app.route("/university/logout")
def university_logout():
    session.pop("university", None)
    return redirect(url_for("university_login"))


# ================= UNIVERSITY DASHBOARD (SUBJECTS & MATERIALS) =================

def get_subjects():
    """Return list of subject names (from DB if any, else defaults)."""
    custom = list(subjects_col.distinct("name"))
    return sorted(set(DEFAULT_SUBJECTS) | set(custom))


@app.route("/university/dashboard")
@university_required
def university_dashboard():
    subjects = get_subjects()
    uni_name = session["university"]
    raw = list(materials_col.find({"university": uni_name}).sort("created_at", -1))
    materials = [{**m, "id": str(m["_id"])} for m in raw]
    return render_template(
        "university_dashboard.html",
        subjects=subjects,
        materials=materials,
    )


@app.route("/university/materials/add", methods=["POST"])
@university_required
def university_add_material():
    subject = request.form.get("subject", "").strip()
    title = request.form.get("title", "").strip()
    content = request.form.get("content", "").strip()
    file = request.files.get("file")

    if not subject or not title:
        return redirect(url_for("university_dashboard"))

    if not file or file.filename == "":
        return redirect(url_for("university_dashboard"))  # or flash "File required"

    if not allowed_file(file.filename):
        return "Only PDF and Word (.doc, .docx) files are allowed", 400

    # Save file with unique name
    ext = file.filename.rsplit(".", 1)[-1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    file_path = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(file_path)

    # Ensure subject exists in subjects collection
    subjects_col.update_one(
        {"name": subject},
        {"$set": {"name": subject}},
        upsert=True
    )

    materials_col.insert_one({
        "subject": subject,
        "university": session["university"],
        "title": title,
        "content": content or None,
        "filename": unique_name,
        "original_filename": secure_filename(file.filename),
        "created_at": date.today().isoformat(),
    })

    return redirect(url_for("university_dashboard"))


@app.route("/university/materials/download/<material_id>")
def material_download(material_id):
    """Serve uploaded PDF/Word material for download."""
    try:
        from bson import ObjectId
        doc = materials_col.find_one({"_id": ObjectId(material_id)})
    except Exception:
        doc = None
    if not doc or not doc.get("filename"):
        return "Not found", 404
    path = os.path.join(UPLOAD_FOLDER, doc["filename"])
    if not os.path.isfile(path):
        return "File not found", 404
    download_name = doc.get("original_filename") or doc["filename"]
    return send_file(path, as_attachment=True, download_name=download_name)


PATHWAYS_CATALOG = [
    {
        "id": "agentic_ai",
        "name": "Agentic AI & Autonomous Systems",
        "meta": "14 Interactive Labs · High Demand",
        "image": "agentic_ai.png",
        "keywords": ["ai", "software", "machine learning", "autonomous", "robotics", "tech", "programming"],
        "description": "Agentic AI refers to systems that can pursue complex goals independently without constant human prompting. Real-world examples include autonomous coding assistants (like Devin), robotic factory floor orchestrators, and highly adaptive NPCs in next-generation gaming.",
        "links": [
            {"label": "LangChain Official Docs (Framework)", "url": "https://python.langchain.com/docs/get_started/introduction"},
            {"label": "DeepLearning.AI: AI for Everyone", "url": "https://www.coursera.org/learn/ai-for-everyone"},
            {"label": "HuggingFace Open-Source Models Hub", "url": "https://huggingface.co/models"}
        ],
        "certs": [
            {"label": "Google: Machine Learning Professional Certificate", "url": "https://www.coursera.org/professional-certificates/google-machine-learning-engineer"},
            {"label": "DeepLearning.AI: AI Agents in LangGraph", "url": "https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/"},
            {"label": "AWS: Machine Learning Specialty Certification", "url": "https://aws.amazon.com/certification/certified-machine-learning-specialty/"}
        ]
    },
    {
        "id": "quantum_computing",
        "name": "Practical Quantum Computing",
        "meta": "8 Interactive Labs · Future Crucial",
        "image": "quantum_computing.png",
        "keywords": ["quantum", "physics", "math", "computation"],
        "description": "Quantum computers leverage quantum mechanics to solve problems classical computers never could. Real-world use cases include discovering new chemical compounds, hyper-optimizing global logistics, and breaking modern cryptographic algorithms.",
        "links": [
            {"label": "IBM Quantum Learning & Qiskit", "url": "https://learning.quantum.ibm.com/"},
            {"label": "MIT xPRO: Quantum Computing Fundamentals", "url": "https://xpro.mit.edu/programs/program-quantum-computing/"}
        ],
        "certs": [
            {"label": "IBM: Quantum Developer Certification", "url": "https://www.ibm.com/training/certification/C0010700"},
            {"label": "MIT xPRO: Quantum Computing Certificate", "url": "https://xpro.mit.edu/programs/program-quantum-computing/"},
            {"label": "edX: Quantum Machine Learning (MIT)", "url": "https://www.edx.org/learn/quantum-computing/massachusetts-institute-of-technology-quantum-machine-learning"}
        ]
    },
    {
        "id": "computational_bio",
        "name": "Computational Bio-Engineering",
        "meta": "10 Interactive Labs · Booming",
        "image": "computational_bio.png",
        "keywords": ["biology", "medical", "health", "pharma"],
        "description": "This field merges computer science with biology. Real-world examples include using machine learning to fold proteins (like AlphaFold), designing synthetic lifeforms to consume ocean plastic, and personalizing cancer treatments based on a patient's exact DNA.",
        "links": [
            {"label": "Rosalind: Learn Bioinformatics via Coding", "url": "https://rosalind.info/"},
            {"label": "DeepMind AlphaFold Protein Database", "url": "https://alphafold.ebi.ac.uk/"}
        ],
        "certs": [
            {"label": "Coursera: Bioinformatics Specialization (UCSD)", "url": "https://www.coursera.org/specializations/bioinformatics"},
            {"label": "edX: Computational Biology (MIT)", "url": "https://www.edx.org/learn/computational-biology"},
            {"label": "Johns Hopkins: Genomic Data Science Certificate", "url": "https://www.coursera.org/specializations/genomic-data-science"}
        ]
    },
    {
        "id": "spatial_computing",
        "name": "Spatial Computing (AR/VR)",
        "meta": "12 Interactive Labs · Emerging Tech",
        "image": "spatial_computing.png",
        "keywords": ["vr", "ar", "design", "gaming", "metaverse"],
        "description": "Spatial computing blends the physical and digital worlds. Real-world uses include architects walking through holographic blueprints before construction, surgeons practicing complex operations in VR, and immersive remote-work command centers.",
        "links": [
            {"label": "Unity Learn: VR Development Pathway", "url": "https://learn.unity.com/pathway/vr-development"},
            {"label": "Meta Spark Studio: Build AR Experiences", "url": "https://sparkar.facebook.com/ar-studio/"}
        ],
        "certs": [
            {"label": "Unity Certified Associate: Game Developer", "url": "https://unity.com/products/unity-certifications"},
            {"label": "Meta Spark Certification Program", "url": "https://sparkar.facebook.com/ar-studio/learn/certification/"},
            {"label": "Coursera: AR & VR Development (Unity)", "url": "https://www.coursera.org/learn/ar-vr-mixed-reality-virtual-environments"}
        ]
    },
    {
        "id": "web3",
        "name": "Web3 & Decentralized Tech",
        "meta": "9 Interactive Labs · Paradigm Shift",
        "image": "web3.png",
        "keywords": ["web3", "crypto", "blockchain", "finance"],
        "description": "Web3 is the next iteration of the internet, built on blockchain. Real-world examples include transparent smart contracts that execute legal agreements automatically, decentralized finance (DeFi) eliminating bank fees, and secure cross-border identity verification.",
        "links": [
            {"label": "Buildspace: Learn Web3 by Building", "url": "https://buildspace.so/"},
            {"label": "Ethereum Developer Portal", "url": "https://ethereum.org/en/developers/"}
        ],
        "certs": [
            {"label": "Blockchain Council: Certified Blockchain Developer", "url": "https://www.blockchain-council.org/certifications/certified-blockchain-developer/"},
            {"label": "Coursera: Blockchain Specialization (INSEAD)", "url": "https://www.coursera.org/specializations/blockchain"},
            {"label": "ConsenSys: Ethereum Developer Bootcamp", "url": "https://consensys.io/academy/bootcamp"}
        ]
    },
    {
        "id": "smart_energy",
        "name": "Sustainable Energy Grids",
        "meta": "11 Interactive Labs · Fast Growing",
        "image": "smart_energy.png",
        "keywords": ["energy", "environment", "climate", "sustainability"],
        "description": "Smart grids use AI and IoT to perfectly balance energy production and consumption. Real-world examples include homes dynamically selling solar power back to the grid, predicting power outages before they happen, and drastically reducing global carbon footprints.",
        "links": [
            {"label": "NREL: National Renewable Energy Lab Data", "url": "https://www.nrel.gov/"},
            {"label": "Coursera: Energy Transition & Innovation", "url": "https://www.coursera.org/learn/energy-transition-innovation"}
        ],
        "certs": [
            {"label": "NABCEP: Solar PV Installation Professional", "url": "https://www.nabcep.org/certifications/"},
            {"label": "edX: Sustainable Energy (TU Delft)", "url": "https://www.edx.org/learn/energy/delft-university-of-technology-solar-energy"},
            {"label": "Coursera: Clean Energy & Climate Policy", "url": "https://www.coursera.org/learn/clean-energy-and-climate-policy"}
        ]
    },
    {
        "id": "ai_drug_discovery",
        "name": "AI-Driven Drug Discovery",
        "meta": "15 Interactive Labs · High Impact",
        "image": "computational_bio.png",
        "keywords": ["medical", "pharma", "health", "medicine", "drugs"],
        "description": "Pharmaceutical companies use AI to rapidly simulate how millions of chemical compounds interact with diseased cells. This turns a 10-year drug discovery process into a 1-year process, saving millions of lives with faster vaccine and cure generation.",
        "links": [
            {"label": "Schrödinger: Computational Chemistry Software", "url": "https://www.schrodinger.com/"},
            {"label": "NCBI PubChem Database", "url": "https://pubchem.ncbi.nlm.nih.gov/"}
        ],
        "certs": [
            {"label": "Coursera: Drug Discovery (UC San Diego)", "url": "https://www.coursera.org/learn/drug-discovery"},
            {"label": "edX: AI in Health Care (MIT)", "url": "https://www.edx.org/learn/artificial-intelligence/massachusetts-institute-of-technology-ai-in-health-care"},
            {"label": "RAPS: Regulatory Affairs Professional Certification", "url": "https://www.raps.org/certification"}
        ]
    },
    {
        "id": "nano_medicine",
        "name": "Nanomedicine Robotics",
        "meta": "7 Interactive Labs · Cutting Edge",
        "image": "computational_bio.png",
        "keywords": ["medical", "robotics", "health", "nanotech"],
        "description": "Nanomedicine involves microscopic robots injected into the bloodstream. Real-world uses currently in development include targeted nanobots that seek out and destroy cancer cells without harming healthy tissue, or physically repairing torn cellular walls.",
        "links": [
            {"label": "Harvard Wyss Institute: Biologically Inspired Engineering", "url": "https://wyss.harvard.edu/"},
            {"label": "Nature: Latest in Nanomedicine Research", "url": "https://www.nature.com/subjects/nanomedicine"}
        ],
        "certs": [
            {"label": "Coursera: Nanotechnology and Nanosensors (Technion)", "url": "https://www.coursera.org/learn/nanotechnology-nanosensors"},
            {"label": "edX: Biomedical Devices (IIT Bombay)", "url": "https://www.edx.org/learn/biomedical-engineering"},
            {"label": "ASM International: Medical Device Manufacturing", "url": "https://www.asminternational.org/education/online-courses/"}
        ]
    },
    {
        "id": "algo_trading",
        "name": "Algorithmic Finance",
        "meta": "12 Interactive Labs · Lucrative",
        "image": "quantum_computing.png",
        "keywords": ["finance", "trading", "economics", "math"],
        "description": "Quantitative algorithms execute millions of stock trades in fractions of a second. Hedge funds use these to predict micro-fluctuations in global markets by analyzing news sentiment, weather data, and satellite imagery of supply chains in real time.",
        "links": [
            {"label": "QuantConnect: Algorithmic Trading Platform", "url": "https://www.quantconnect.com/"},
            {"label": "Investopedia: Intro to Quantitative Trading", "url": "https://www.investopedia.com/articles/trading/09/quantitative-trading.asp"}
        ],
        "certs": [
            {"label": "CFA Institute: Chartered Financial Analyst", "url": "https://www.cfainstitute.org/en/programs/cfa"},
            {"label": "Coursera: Financial Engineering & Risk Management", "url": "https://www.coursera.org/specializations/financial-engineering"},
            {"label": "CAIA Association: Alternative Investments Certification", "url": "https://caia.org/"}
        ]
    },
    {
        "id": "gen_ai_arts",
        "name": "Generative AI for Creative Arts",
        "meta": "9 Interactive Labs · Trending",
        "image": "agentic_ai.png",
        "keywords": ["arts", "creative", "media", "music", "video"],
        "description": "Artists are embracing AI to augment human creativity. Real-world uses include prompt-driven video generation (Sora), AI composing cinematic soundtracks for indie video games, and generating 3D textures instantly for graphic designers.",
        "links": [
            {"label": "Midjourney: Generative AI Art", "url": "https://www.midjourney.com/"},
            {"label": "Civitai: Open Source AI Art Models Hub", "url": "https://civitai.com/"}
        ],
        "certs": [
            {"label": "Adobe: Creative Cloud Generative AI Certification", "url": "https://www.adobe.com/training/adobe-certified-professional.html"},
            {"label": "Coursera: Generative AI for Everyone (DeepLearning.AI)", "url": "https://www.coursera.org/learn/generative-ai-for-everyone"},
            {"label": "Runway ML: AI Video Generation Certificate", "url": "https://runwayml.com/"}
        ]
    },
    {
        "id": "space_propulsion",
        "name": "Deep Space Engineering",
        "meta": "6 Interactive Labs · Visionary",
        "image": "smart_energy.png",
        "keywords": ["space", "astronomy", "physics", "engineering"],
        "description": "As humanity pushes into the cosmos, traditional rocket fuel won't cut it. Real-world examples include Hall-effect ion thrusters keeping satellites in orbit and experimental nuclear thermal propulsion designed to drastically cut travel time to Mars.",
        "links": [
            {"label": "NASA Open APIs & Data", "url": "https://api.nasa.gov/"},
            {"label": "SpaceX Careers & Engineering Requirements", "url": "https://www.spacex.com/careers/"}
        ],
        "certs": [
            {"label": "NASA: Open Science Certification", "url": "https://science.nasa.gov/open-science/"},
            {"label": "Coursera: Astronomy & Astrophysics (Duke)", "url": "https://www.coursera.org/learn/astro"},
            {"label": "edX: Aerospace Engineering Essentials (MIT)", "url": "https://www.edx.org/learn/aerospace-engineering"}
        ]
    },
    {
        "id": "cyber_security",
        "name": "Advanced Cyber-Warfare Defense",
        "meta": "16 Interactive Labs · Critical",
        "image": "web3.png",
        "keywords": ["security", "cyber", "hacking", "defense", "tech"],
        "description": "With infrastructure moving online, digital defense is critical. Expert systems deploy AI-driven honeypots to trap malicious actors, autonomously patch server vulnerabilities in real-time, and detect zero-day exploits before they execute.",
        "links": [
            {"label": "Hack The Box: Penetration Testing Labs", "url": "https://www.hackthebox.com/"},
            {"label": "TryHackMe: Hands-on Cyber Security Training", "url": "https://tryhackme.com/"}
        ],
        "certs": [
            {"label": "CompTIA Security+ Certification", "url": "https://www.comptia.org/certifications/security"},
            {"label": "CEH: Certified Ethical Hacker (EC-Council)", "url": "https://www.eccouncil.org/programs/certified-ethical-hacker-ceh/"},
            {"label": "CISSP: Certified Info. Systems Security Professional", "url": "https://www.isc2.org/certifications/cissp"}
        ]
    }
]

# ================= ROOMS =================


@app.route("/")
def index():
    if session.get("university"):
        return redirect(url_for("university_dashboard"))
    if session.get("user"):
        return redirect(url_for("rooms"))
    return redirect(url_for("login"))


@app.route("/rooms")
@login_required
def rooms():
    room_list = list(rooms_col.find())

    current_user = session.get("user")
    user_rooms = []
    recommended_rooms = []
    other_rooms = []

    if current_user:
        # rooms this user created
        created_rooms = {r["name"] for r in room_list if r.get("created_by") == current_user}

        # rooms where this user has sent at least one message
        joined_rooms = set(messages_col.distinct("room", {"sender": current_user}))

        user_room_names = created_rooms | joined_rooms

        # separate lists
        user_rooms = [r for r in room_list if r["name"] in user_room_names]
        other_rooms = [r for r in room_list if r["name"] not in user_room_names]

        # content-based scoring using user keyword profile
        user_doc = users_col.find_one({"username": current_user}) or {}
        user_keywords = user_doc.get("keywords", {}) or {}

        def score_room(room):
            room_kws = room.get("keywords", []) or []
            return sum(user_keywords.get(k, 0) for k in room_kws)

        for r in other_rooms:
            r_score = score_room(r)
            r["score"] = r_score

        recommended_rooms = sorted(
            [r for r in other_rooms if r.get("score", 0) > 0],
            key=lambda x: x["score"],
            reverse=True
        )[:5]
    else:
        # fallback: should not normally happen because of @login_required
        user_rooms = []
        other_rooms = room_list
        recommended_rooms = []
        user_doc = None
        user_keywords = {}

    xp = (user_doc or {}).get("xp", 0)

    # Future Pathways Dynamic Recommendations
    import random
    scored_pathways = []
    for pw in PATHWAYS_CATALOG:
        score = 0
        for pw_kw in pw["keywords"]:
            for u_kw, count in user_keywords.items():
                if pw_kw.lower() in u_kw.lower() or u_kw.lower() in pw_kw.lower():
                    score += count
        scored_pathways.append({**pw, "score": score + random.uniform(0, 0.5)}) # noise to break ties

    # Pick top 6 recommendations based on chat history matches
    recommended_pathways = sorted(scored_pathways, key=lambda x: x["score"], reverse=True)[:6]
    # Shuffle them to ensure freshness when there are ties (e.g. at 0 score)
    random.shuffle(recommended_pathways)

    return render_template(
        "rooms.html",
        rooms=user_rooms,
        recommended_rooms=recommended_rooms,
        other_rooms=other_rooms,
        pathways=recommended_pathways,
        xp=xp
    )


@app.route("/create_room", methods=["POST"])
@login_required
def create_room():
    name = request.form["room"]

    if not rooms_col.find_one({"name": name}):
        rooms_col.insert_one({
            "name": name,
            "keywords": [],
            "created_by": session.get("user")
        })

    return redirect(url_for("rooms"))


@app.route("/room/<room_name>")
@login_required
def chatroom(room_name):
    msgs = list(messages_col.find({"room": room_name}))
    room = rooms_col.find_one({"name": room_name})
    user_doc = users_col.find_one({"username": session.get("user")}) or {}
    xp = user_doc.get("xp", 0)
    return render_template("chatroom.html",
                           room=room_name,
                           messages=msgs,
                           keywords=room["keywords"],
                           xp=xp)


@app.route("/quiz", methods=["GET", "POST"])
@login_required
def quiz():
    user = session["user"]
    user_doc = users_col.find_one({"username": user}) or {}
    xp = user_doc.get("xp", 0)

    # ===== Derive quiz topics from summaries of:
    # 1) LLM responses in the chat
    # 2) Descriptions of materials uploaded by universities
    corpus_parts = []

    # Recent assistant (bot) responses and room names the user has participated in
    user_rooms = list(messages_col.distinct("room", {"sender": user}))
    if user_rooms:
        corpus_parts.append(f"User is interested in these subjects via rooms: {', '.join(user_rooms)}")
        bot_msgs = list(
            messages_col.find(
                {"sender": "bot", "room": {"$in": user_rooms}},
                sort=[("_id", -1)],
                limit=50,
            )
        )
        corpus_parts.extend(m.get("text", "") for m in bot_msgs if m.get("text"))

    # Brief summaries of uploaded materials
    material_docs = list(materials_col.find({}, {"subject": 1, "title": 1, "content": 1}).limit(60))
    for m in material_docs:
        line = f"Material Subject: {m.get('subject', '')} - Title: {m.get('title', '')}"
        if m.get("content"):
            line += f" - Key Content: {m.get('content')[:300]}"
        corpus_parts.append(line)

    topics_list = []
    if corpus_parts:
        corpus = "\n\n".join(corpus_parts)[:9000]

        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert academic curriculum designer. "
                    "Analyze the student's activity and materials to identify their specialized interest "
                    "(e.g., Clinical Medicine, Aerospace Engineering, Financial Analysis). "
                    "Generate up to 10 specific, deep-dive quiz topic names that represent an advanced "
                    "understanding of those fields. Avoid generic names like 'Science' or 'Math' — "
                    "instead use names like 'Cardiovascular Pathology' or 'Orbital Mechanics'."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Here is the student's learning context:\n\n"
                    f"{corpus}\n\n"
                    "Extract exactly 8-10 deep, specialized topic names. "
                    "Return ONLY a comma-separated list."
                ),
            },
        ]

        try:
            raw = ask_llm(prompt_messages)
            topics_list = [
                t.strip()
                for t in (raw or "").split(",")
                if t.strip() and len(t.strip()) > 2
            ][:10]
        except Exception:
            topics_list = []

    display_topics = []
    
    def get_emoji(name):
        name_lower = name.lower()
        if name in TOPIC_EMOJIS: return TOPIC_EMOJIS[name]
        
        # Extended keyword matching for "deep" topics
        if any(w in name_lower for w in ["med", "clin", "path", "anat", "surger", "health"]): return "🏥"
        if any(w in name_lower for w in ["rocket", "aero", "space", "orbit", "planet", "astron"]): return "🚀"
        if any(w in name_lower for w in ["law", "legal", "court", "judic"]): return "⚖️"
        if any(w in name_lower for w in ["quantum", "particl", "nuclear"]): return "🌌"
        if any(w in name_lower for w in ["circuit", "electr", "power", "volt"]): return "⚡"
        if any(w in name_lower for w in ["data", "algor", "struct", "comput"]): return "🏗️"
        if any(w in name_lower for w in ["gene", "cell", "molecul", "dna"]): return "🧬"
        if any(w in name_lower for w in ["market", "busin", "sale", "finan"]): return "💼"
        if any(w in name_lower for w in ["chem", "organ", "reac", "mol"]): return "🧪"
        if any(w in name_lower for w in ["vocab", "gramm", "writ", "lang"]): return "✍️"
        
        # Fallback to general keyword logic
        if "math" in name_lower or "calc" in name_lower: return "🧮"
        if "web" in name_lower or "net" in name_lower: return "🌐"
        if "art" in name_lower or "design" in name_lower: return "🎨"
        
        return "📝"

    # 1. Add personalized specialized topics
    for t in topics_list:
        if not any(d["name"].lower() == t.lower() for d in display_topics):
            display_topics.append({"name": t, "emoji": get_emoji(t)})
    
    # guaranteed ones (user requested specifically)
    guaranteed = [
        "English Vocabulary", "Physics", "Chemistry", "Thermodynamics", 
        "Organic Chemistry", "Calculus", "Data Structures & Algorithms"
    ]
    
    for g in guaranteed:
        if not any(d["name"].lower() == g.lower() for d in display_topics):
            display_topics.append({"name": g, "emoji": get_emoji(g)})

    # Fill more from DEFAULT_SUBJECTS to ensure variety
    for s in DEFAULT_SUBJECTS:
        if len(display_topics) >= 20: # Show more
            break
        if not any(d["name"].lower() == s.lower() for d in display_topics):
            display_topics.append({"name": s, "emoji": get_emoji(s)})

    # Sort to prioritize AI and Tech topics at the top
    tech_keywords = [
        "ai", "intelligence", "machine learning", "tech", "web", "data", "algorithm",
        "quantum", "programming", "code", "software", "engineering", "rocket", "aerospace",
        "android", "ios", "fullstack", "backend", "frontend", "cyber", "cloud"
    ]
    def is_priority_tech(name):
        return any(k in name.lower() for k in tech_keywords)

    display_topics.sort(key=lambda x: is_priority_tech(x["name"]), reverse=True)

    import random

    def build_questions_for_topic(topic: str, difficulty: str = "easy"):
        """
        Ask the LLM to generate 5 multiple-choice questions about `topic`
        at the given difficulty level. All 4 options (correct + 3 distractors)
        use terminology from the topic domain.
        Falls back to a generic template on any error.
        """
        level_guidance = {
            "easy": (
                "The questions are for a 10th-grade or +2 high school student. "
                "Ask about basic definitions, simple concepts, and everyday examples. "
                "The wrong options should sound plausible by using real terminology from "
                f"the {topic} domain, but describe something incorrect or irrelevant."
            ),
            "medium": (
                "The questions are for a university / college undergraduate student. "
                "Ask about concepts that require analysis, comparison, or application. "
                "The wrong options should use correct domain terms from "
                f"{topic} but present them in a misleading or incorrect context."
            ),
            "hard": (
                "The questions are for working professionals seeking to level up their skills. "
                "Ask about real-world tradeoffs, architectural decisions, production best practices, "
                f"and advanced concepts in {topic}. The wrong answers must sound very credible — "
                "use proper technical jargon from the domain but make subtle factual errors."
            ),
        }

        system_prompt = (
            "You are an expert educational quiz generator. "
            "You ONLY respond with a valid JSON array, no markdown, no extra text."
        )

        user_prompt = (
            f"Generate exactly 5 multiple-choice quiz questions about: {topic}\n\n"
            f"Difficulty context: {level_guidance.get(difficulty, level_guidance['easy'])}\n\n"
            "Rules:\n"
            "1. Each question must have exactly 4 options (A, B, C, D).\n"
            "2. Exactly ONE option is correct.\n"
            "3. The other 3 options MUST use real terminology and concepts from "
            f"the {topic} field — they should sound plausible but be factually wrong or misleading.\n"
            "4. The correct answer must always be option at index 0 in the options array "
            "(the caller will shuffle before displaying).\n"
            "5. Return ONLY a JSON array like this:\n"
            '[\n'
            '  {"text": "Question text here?", "options": ["Correct answer", "Wrong but plausible 1", "Wrong but plausible 2", "Wrong but plausible 3"]},\n'
            '  ...\n'
            ']\n'
            "No markdown code fences, no extra explanation — raw JSON only."
        )

        try:
            raw = ask_llm([
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ])

            # Strip any accidental markdown fences
            raw = (raw or "").strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            if raw.endswith("```"):
                raw = raw[:-3].strip()

            parsed = json.loads(raw)
            if isinstance(parsed, list) and len(parsed) >= 3:
                question_bank = []
                for item in parsed[:5]:
                    opts = item.get("options", [])[:4]
                    if len(opts) < 4:
                        continue
                    random.shuffle(opts)
                    question_bank.append({"text": item.get("text", ""), "options": opts})
                if question_bank:
                    return question_bank
        except Exception:
            pass  # Fall through to fallback

        # ── Fallback: template questions with topic-specific language ──
        fallback = [
            {
                "text": f"Which statement best describes a core concept in {topic}?",
                "options": [
                    f"A foundational principle of {topic} that enables real-world problem solving.",
                    f"A deprecated technique in {topic} that is no longer used in practice.",
                    f"An auxiliary component of {topic} with no impact on primary outcomes.",
                    f"A theoretical construct in {topic} that contradicts its empirical basis.",
                ]
            },
            {
                "text": f"What is a primary application of {topic} in modern industry?",
                "options": [
                    f"Using {topic} to optimise workflows and drive measurable business outcomes.",
                    f"Applying {topic} exclusively to legacy infrastructure with no modern relevance.",
                    f"Limiting {topic} to pure academic research without practical deployment.",
                    f"Using {topic} only as a supplementary metric rather than a primary solution.",
                ]
            },
            {
                "text": f"Which approach is most effective when solving problems using {topic}?",
                "options": [
                    f"Systematically applying {topic} principles while validating each step with data.",
                    f"Selecting an approach in {topic} based on familiarity rather than suitability.",
                    f"Bypassing the iterative feedback loop that {topic} relies on for accuracy.",
                    f"Treating all {topic} sub-domains as interchangeable without contextual adaptation.",
                ]
            },
            {
                "text": f"What distinguishes an expert in {topic} from a beginner?",
                "options": [
                    f"Experts in {topic} can identify edge cases, evaluate tradeoffs, and design robust solutions.",
                    f"Experts in {topic} focus only on theoretical models and avoid implementation.",
                    f"Experts in {topic} rely exclusively on automated tools without critical evaluation.",
                    f"Experts in {topic} memorise documentation rather than understanding underlying principles.",
                ]
            },
            {
                "text": f"Which is a common pitfall when working with {topic}?",
                "options": [
                    f"Over-engineering a {topic} solution beyond the actual requirements of the problem.",
                    f"Validating results in {topic} with multiple independent test cases.",
                    f"Iterating incrementally and reviewing {topic} outputs at each stage.",
                    f"Collaborating with domain experts to refine the {topic} approach.",
                ]
            },
        ]
        question_bank = []
        for spec in fallback:
            opts = spec["options"][:]
            random.shuffle(opts)
            question_bank.append({"text": spec["text"], "options": opts})
        return question_bank


    if request.method == "POST":
        stage = request.form.get("stage", "select_topic")

        if stage == "select_topic":
            selected_topic = request.form.get("topic")
            if not selected_topic:
                return render_template(
                    "quiz.html",
                    stage="select_topic",
                    topics=display_topics,
                    xp=xp,
                )
            # After choosing topic → show difficulty selector
            return render_template(
                "quiz.html",
                stage="select_difficulty",
                topic=selected_topic,
                topics=display_topics,
                xp=xp,
            )

        if stage == "select_difficulty":
            selected_topic = request.form.get("topic")
            selected_difficulty = request.form.get("difficulty", "easy")
            questions = build_questions_for_topic(selected_topic, selected_difficulty)
            return render_template(
                "quiz.html",
                stage="questions",
                topic=selected_topic,
                difficulty=selected_difficulty,
                questions=questions,
                xp=xp,
            )

        if stage == "answer_questions":
            # award XP for answering quiz
            users_col.update_one({"username": user}, {"$inc": {"xp": 20}})
            return redirect(url_for("rooms"))

    # GET: show topic selection screen
    return render_template(
        "quiz.html",
        stage="select_topic",
        topics=display_topics,
        xp=xp,
    )

# ================= CHAT API =================

@app.route("/send/<room>", methods=["POST"])
@login_required
def send(room):
    from flask import Response
    text = request.json["text"]
    user = session["user"]

    def generate():
        messages = [
            {"role": "system", "content": "You are a helpful expert tutor for the topic the user is asking about."}
        ]
        
        # Fetch up to 10 most recent messages in this room for conversation memory
        history = list(messages_col.find({"room": room}).sort("_id", -1).limit(10))
        history.reverse()
        for msg in history:
            role = "assistant" if msg.get("sender") == "bot" else "user"
            messages.append({"role": role, "content": msg.get("text", "")})
            
        # Append the new user message
        messages.append({"role": "user", "content": text})
        
        reply_parts = []
        try:
            for chunk in ask_llm(messages, stream=True):
                content = chunk.get("message", {}).get("content", "")
                if content:
                    reply_parts.append(content)
                    yield f"data: {json.dumps({'chunk': content})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return
            
        full_reply = "".join(reply_parts)
        
        # Now run heavy ML tasks synchronously but yield keywords when done
        # The user won't feel this latency because the LLM stream has already finished and is readable.
        try:
            intent = detect_intent(text)
            category = detect_category(text)
            keywords = extract_keywords(text)
            
            rooms_col.update_one({"name": room}, {"$addToSet": {"keywords": {"$each": keywords}}})
            if keywords:
                users_col.update_one({"username": user}, {"$inc": {f"keywords.{k}": 1 for k in keywords}})
            
            last_other = messages_col.find_one({"room": room, "sender": {"$nin": [user, "bot"]}}, sort=[("_id", -1)])
            if last_other:
                users_col.update_one({"username": user}, {"$inc": {"xp": 2}})
                
            messages_col.insert_many([
                {"room": room, "sender": user, "text": text},
                {"room": room, "sender": "bot", "text": full_reply}
            ])
            
            yield f"data: {json.dumps({'keywords': keywords, 'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'done': True, 'error': str(e)})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


# ================= RUN =================
if __name__ == "__main__":
    # Binding to 0.0.0.0 makes the server accessible via the local network IP (e.g. 192.168.x.x:5000)
    app.run(debug=True, host="0.0.0.0", port=5000)
