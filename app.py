import os
import uuid
import datetime
import streamlit as st
import chromadb
from google import genai
from google.genai import types
import gspread
from google.oauth2.service_account import Credentials

# --- 1. CONFIGURATION & CLIENT SETUP ---
st.set_page_config(page_title="Multi-Bank RAG Assistant", layout="wide")

# Initialize Gemini Client
api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
if not api_key:
    st.error("Error: No API Key")
    st.stop()

ai_client = genai.Client(api_key=api_key)

# Initialize ChromaDB persistent client
@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path="./chroma_db")

chroma_client = get_chroma_client()

# --- 2. GOOGLE SHEETS CONNECTION SETUP ---
@st.cache_resource
def get_gspread_sheet():
    try:
        # Pulls service account dictionary directly from Streamlit secrets
        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        
        # Load credentials from st.secrets
        if "gcp_service_account" in st.secrets:
            creds_dict = dict(st.secrets["gcp_service_account"])
            creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
            gc = gspread.authorize(creds)
            
            # Open the Google Sheet by name (Ensure you share this sheet with your service account email)
            sheet_name = st.secrets.get("GOOGLE_SHEET_NAME", "Bank_RAG_Audit_Database")
            sheet = gc.open(sheet_name).sheet1
            return sheet
        else:
            return None
    except Exception as e:
        st.sidebar.error(f"Google Sheets Connection Warning: {e}")
        return None

sheet_db = get_gspread_sheet()

# --- 3. MAPPINGS & CONFIGS ---
BANK_CONFIGS = {
    "bofa": {
        "display_name": "Bank of America",
        "primary": "#012169",  # Navy Blue
        "accent": "#E31837"    # Crimson Red
    },
    "usbank": {
        "display_name": "U.S. Bank",
        "primary": "#0C2074",  # Sapphire Blue
        "accent": "#D9282E"    # Cherry Red
    },
    "chase": {
        "display_name": "Chase",
        "primary": "#117ACA",  # Chase Blue
        "accent": "#211E1E"    # Dark Onyx
    },
    "wellsfargo": {
        "display_name": "Wells Fargo",
        "primary": "#BB0826",  # Wells Fargo Red
        "accent": "#FCC60A"    # Stagecoach Yellow
    }
}

def get_bank_config(col_name):
    clean_key = col_name.replace("doc_", "").lower()
    for key, cfg in BANK_CONFIGS.items():
        if key in clean_key:
            return cfg
    formatted = col_name.replace("doc_", "").replace("_", " ").title()
    return {"display_name": formatted, "primary": "#1f77b4", "accent": "#ff7f0e"}

# --- 4. SIDEBAR: BANK SELECTION & STYLING ---
st.sidebar.title("🏦 Bank Selection")

try:
    collections = chroma_client.list_collections()
    collection_list = [col.name for col in collections]
except Exception:
    collection_list = []

if not collection_list:
    st.sidebar.warning("No docs found in ChromaDB, run chunks")
    selected_col = None
    bank_cfg = {"display_name": "None", "primary": "#111111", "accent": "#888888"}
else:
    col_display_map = {col: get_bank_config(col)["display_name"] for col in collection_list}
    selected_col = st.sidebar.selectbox("Select Institution:", list(col_display_map.keys()), format_func=lambda x: col_display_map[x])
    bank_cfg = get_bank_config(selected_col)

# --- APPLY SIDEBAR SPLIT GRADIENT (Top = Accent, Bottom = Primary) ---
st.markdown(
    f"""
    <style>
        [data-testid="stSidebar"] {{
            background: linear-gradient(
                to bottom, 
                {bank_cfg['accent']} 0%, 
                {bank_cfg['accent']} 50%, 
                {bank_cfg['primary']} 50%, 
                {bank_cfg['primary']} 100%
            ) !important;
            color: #FFFFFF !important;
        }}
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3, 
        [data-testid="stSidebar"] label, 
        [data-testid="stSidebar"] .stMarkdown {{
            color: #FFFFFF !important;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

# --- UNIQUE RESPONSE CODE LOOKUP SECTION ---
st.sidebar.subheader("🔍 Response Inspector")
lookup_code = st.sidebar.text_input("Enter Response Code:", placeholder="e.g., RAG-A1B2C3")
if lookup_code:
    if "response_metadata_store" in st.session_state and lookup_code.strip() in st.session_state.response_metadata_store:
        st.sidebar.success("Found response receipt")
        meta_record = st.session_state.response_metadata_store[lookup_code.strip()]
        with st.sidebar.expander("View Details", expanded=True):
            st.json(meta_record)
    else:
        st.sidebar.error("Code not found in session history.")

st.sidebar.markdown("---")
st.sidebar.info(
    "**Note:**\n"
    "Powered by Gemini's free tier. Might be slow, might stop working."
)

# --- 5. MAIN INTERFACE & ONBOARDING STARTER QUESTIONS ---
st.title(f"📄 {bank_cfg['display_name']} Agreement Assistant")

with st.expander("Getting started:", expanded=True):
    st.markdown(
        """
        This tool searches official **Bank Deposit Agreements and Account Disclosures** using Retrieval-Augmented Generation (RAG). 
        You can ask specific questions regarding fees, requirements, interest tiers, and terms.
        
        **Sample questions you can try:**
        * *What are the monthly maintenance fees and how can they be waived?*
        * *What is the bank's policy regarding overdraft fees and returned items?*
        * *How are interest rates calculated and compounded for relationship tiers?*
        * *What are the requirements for closing an account or disputing a transaction?*
        """
    )

if not selected_col:
    st.warning("Please index at least one bank agreement into ChromaDB to begin.")
    st.stop()

collection = chroma_client.get_collection(name=selected_col)

# Initialize session state stores
if "messages" not in st.session_state:
    st.session_state.messages = []
if "response_metadata_store" not in st.session_state:
    st.session_state.response_metadata_store = {}

# Display chat history
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "response_code" in message:
            code = message["response_code"]
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.caption(f"🔑 Response Code: `{code}` (Paste in sidebar to inspect)")
            with col_b:
                feedback = st.radio("Was this helpful?", ["👍", "👎"], key=f"fb_{code}", label_visibility="collapsed", horizontal=True)
# --- 6. RAG QUERY EXECUTION & GOOGLE SHEETS LOGGING ---
if prompt := st.chat_input(f"Ask a question about {bank_cfg['display_name']}..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching document sections and generating response..."):
            try:
                n_results = 3
                temperature = 0.2
                llm_model = "gemini-3.8-flash"
                embedding_model = "all-MiniLM-L6-v2"

                # Retrieve relevant context from ChromaDB
                results = collection.query(
                    query_texts=[prompt],
                    n_results=n_results,
                    include=["documents", "metadatas", "distances"]
                )
                
                retrieved_docs = results.get("documents", [[]])[0]
                retrieved_metas = results.get("metadatas", [[]])[0]
                retrieved_distances = results.get("distances", [[]])[0] if "distances" in results else [0.0] * len(retrieved_docs)
                
                context_block = ""
                sources_cited = set()
                chunks_referenced_audit = []

                for doc_idx, (doc, meta, dist) in enumerate(zip(retrieved_docs, retrieved_metas, retrieved_distances)):
                    title = meta.get("section_title", "General Section")
                    page = meta.get("start_page", "Unknown")
                    path = meta.get("path", "N/A")
                    
                    context_block += f"\n--- Section: {title} (Page {page}) ---\n{doc}\n"
                    sources_cited.add(f"• **{title}** (Page {page})")
                    
                    chunks_referenced_audit.append({
                        "chunk_index": doc_idx + 1,
                        "section_title": title,
                        "page": page,
                        "toc_path": path,
                        "distance_score": float(dist),
                        "content_snippet": doc[:150] + "..."
                    })

                system_instruction = (
                    "You are a precise, helpful banking assistant. Answer the user's question "
                    "strictly using the provided context excerpts from the banking agreement. "
                    "If the answer cannot be found in the context, state clearly that the document does not cover it. "
                    "Always state the name of the bank, and cite the relevant section title and page number."
                )

                user_prompt = (
                    f"Question: {prompt}\n\n"
                    f"Retrieved Document Context:\n{context_block}"
                )

                response = ai_client.models.generate_content(
                    model=llm_model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=temperature,
                    ),
                )
                
                answer_text = response.text
                
                if sources_cited:
                    answer_text += "\n\n**Sources Referenced:**\n" + "\n".join(sources_cited)

                response_code = f"RAG-{uuid.uuid4().hex[:6].upper()}"
                answer_text += f"\n\n--- \n🔑 *Response ID Code: `{response_code}`*"

                audit_record = {
                    "response_code": response_code,
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "bank_institution": bank_cfg["display_name"],
                    "collection_name": selected_col,
                    "user_query": prompt,
                    "llm_model": llm_model,
                    "embedding_model": embedding_model,
                    "parameters": {
                        "temperature": temperature,
                        "top_k_results": n_results
                    },
                    "assistant_response": answer_text,
                    "chunks_referenced": chunks_referenced_audit,
                    "sources_cited": list(sources_cited)
                }

                st.session_state.response_metadata_store[response_code] = audit_record

                # --- ACTIVE GOOGLE SHEETS WRITE ---
                if sheet_db:
                    try:
                        sheet_db.append_row([
                            audit_record["timestamp"],
                            audit_record["response_code"],
                            audit_record["bank_institution"],
                            audit_record["user_query"],
                            str(audit_record["chunks_referenced"]),
                            "Pending Feedback"
                        ])
                    except Exception as sheet_err:
                        st.sidebar.warning(f"Could not write to Google Sheet: {sheet_err}")

                st.markdown(answer_text)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer_text,
                    "response_code": response_code
                })
                
                st.radio("Was this helpful?", ["👍", "👎"], key=f"fb_{response_code}", label_visibility="collapsed", horizontal=True)

            except Exception as e:
                error_msg = f"**Usage Limit / Error Encountered:** `{str(e)}`\n\nThe free-tier usage limit has been reached or a network error occurred. Please wait a moment before trying again."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
