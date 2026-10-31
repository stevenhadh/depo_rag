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
        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        
        if "gcp_service_account" in st.secrets:
            creds_dict = dict(st.secrets["gcp_service_account"])
            creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
            gc = gspread.authorize(creds)
            
            sheet_name = st.secrets.get("GOOGLE_SHEET_NAME", "depo_rag_receipts")
            sheet = gc.open(sheet_name).sheet1
            return sheet
        else:
            return None
    except Exception as e:
        st.sidebar.error(f"Google Sheets Connection Warning: {e}")
        return None

sheet_db = get_gspread_sheet()

# --- HELPER: ENSURE SHEET HEADERS EXIST ---
def init_sheet_headers(sheet):
    try:
        header = sheet.row_values(1)
        expected_headers = [
            "Timestamp", "Response Code", "Bank Institution", "User Query", 
            "LLM Model", "Embedding Model", "Temperature", "Chunks Referenced", 
            "Assistant Response", "Thumbs Up Count", "Thumbs Down Count"
        ]
        if not header:
            sheet.append_row(expected_headers)
    except Exception:
        pass

if sheet_db:
    init_sheet_headers(sheet_db)

# --- 3. MAPPINGS & CONFIGS ---
BANK_CONFIGS = {
    "bofa": {
        "display_name": "Bank of America",
        "primary": "#E31837",  # Crimson Red
        "accent":  "#012169"   # Navy Blue
    },
    "usbank": {
        "display_name": "U.S. Bank",
        "primary": "#0C2074",  # Sapphire Blue
        "accent": "#D9282E"    # Cherry Red
    },
    "chase": {
        "display_name": "Chase",
        "primary": "#4169E1",  # Royal Blue
        "accent": "#4169E1"    # Royal Blue
    },
    "wellsfargo": {
        "display_name": "Wells Fargo",
        "primary": "#BB0826",  # Wells Fargo Red
        "accent": "#dbb404"    # Dark Gold
    }
}

def get_bank_config(col_name):
    clean_key = col_name.replace("doc_", "").lower()
    for key, cfg in BANK_CONFIGS.items():
        if key in clean_key:
            return cfg
    formatted = col_name.replace("doc_", "").replace("_", " ").title()
    return {"display_name": formatted, "primary": "#1f77b4", "accent": "#ff7f0e"}

# Initialize session state stores
if "messages" not in st.session_state:
    st.session_state.messages = []
if "response_metadata_store" not in st.session_state:
    st.session_state.response_metadata_store = {}
if "user_votes" not in st.session_state:
    st.session_state.user_votes = {}  # Tracks response_code -> "👍" or "👎"

# --- HELPER: SMART VOTE HANDLER (PREVENTS DOUBLE COUNTING) ---
def handle_vote(target_code, new_vote):
    old_vote = st.session_state.user_votes.get(target_code)
    if old_vote == new_vote:
        return  # No change made
    
    # Update local vote state
    st.session_state.user_votes[target_code] = new_vote
    
    # Update Google Sheets counters intelligently using deltas
    if sheet_db:
        try:
            cell = sheet_db.find(target_code)
            if cell:
                row_idx = cell.row
                up_col_idx = 10   # "Thumbs Up Count" column
                down_col_idx = 11 # "Thumbs Down Count" column
                
                current_up = int(sheet_db.cell(row_idx, up_col_idx).value or 0)
                current_down = int(sheet_db.cell(row_idx, down_col_idx).value or 0)
                
                # Revert old vote if it existed
                if old_vote == "👍":
                    current_up = max(0, current_up - 1)
                elif old_vote == "👎":
                    current_down = max(0, current_down - 1)
                
                # Apply new vote
                if new_vote == "👍":
                    current_up += 1
                elif new_vote == "👎":
                    current_down += 1
                
                sheet_db.update_cell(row_idx, up_col_idx, current_up)
                sheet_db.update_cell(row_idx, down_col_idx, current_down)
        except Exception:
            pass

# --- 4. SIDEBAR: BANK SELECTION & STYLING ---
st.sidebar.title("Bank Selection")

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
    selected_col = st.sidebar.selectbox("Select your bank:", list(col_display_map.keys()), format_func=lambda x: col_display_map[x])
    bank_cfg = get_bank_config(selected_col)

# --- APPLY SIDEBAR SPLIT GRADIENT ---
st.markdown(
    f"""
    <style>
        [data-testid="stSidebar"] {{
            background: linear-gradient(
                to bottom, 
                {bank_cfg['accent']} 0%, 
                {bank_cfg['accent']} 33%, 
                {bank_cfg['primary']} 33%, 
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
        
        /* Keep default expander look, but lock the header hover state so it doesn't change color */
        [data-testid="stSidebar"] [data-testid="stExpander"] summary {{
            background-color: transparent !important;
            color: #FFFFFF !important;
        }}
        
        [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {{
            background-color: transparent !important;
            color: #FFFFFF !important;
        }}
        
        [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover p,
        [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover span {{
            color: #FFFFFF !important;
        }}
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

# --- UNIQUE RESPONSE CODE LOOKUP & SIDEBAR VOTING SECTION ---
st.sidebar.subheader("Response Inspector")
lookup_code = st.sidebar.text_input("Enter Response Code:", placeholder="e.g., RAG-A1B2C3")
if lookup_code:
    clean_lookup = lookup_code.strip()
    found_record = None
    
    # 1. Check local session state first
    if clean_lookup in st.session_state.response_metadata_store:
        found_record = st.session_state.response_metadata_store[clean_lookup]
    # 2. Fallback to Google Sheets for cross-instance lookup
    elif sheet_db:
        try:
            records = sheet_db.get_all_records()
            for row in records:
                if str(row.get("Response Code", "")).strip() == clean_lookup:
                    found_record = row
                    break
        except Exception:
            pass

    if found_record:
        # st.sidebar.success("Found response receipt")
        with st.sidebar.expander("View Details", expanded=True):
            st.json(found_record)
            
            # Sidebar Voting Widget
            current_sb_vote = st.session_state.user_votes.get(clean_lookup, None)
            sb_default_idx = 0 if current_sb_vote == "👍" else (1 if current_sb_vote == "👎" else None)
            
            sb_feedback = st.radio(
                "Vote on this response:", 
                ["👍", "👎"], 
                key=f"sb_fb_{clean_lookup}", 
                index=sb_default_idx, 
                horizontal=True
            )
            if sb_feedback and sb_feedback != current_sb_vote:
                handle_vote(clean_lookup, sb_feedback)
                st.rerun()
    else:
        st.sidebar.markdown('<p style="color: #FFFFFF; font-weight: 500; margin-bottom: 0.5rem;">⚠️ Code not found.</p>', unsafe_allow_html=True)

# --- 5. MAIN INTERFACE & ONBOARDING STARTER QUESTIONS ---
st.title(f"{bank_cfg['display_name']} Agreement Assistant")

with st.expander("Getting started:", expanded=True):
    st.markdown(
        """
        This tool searches official **Bank Deposit Agreements** using Retrieval-Augmented Generation (RAG).\n
        You can ask specific questions regarding fees, requirements, and terms.
        
        **Sample questions you can try:**
        * *What documents do I need to open a new account?*
        * *How many days do I have to report an unauthorized charge on my statement?*
        * *Will I be charged a fee if I use an ATM outside of the bank's network?*
        * *If an account holder passes away, what happens to the money?*
        * *How much is the fee if my account goes negative?*

        **Note:**
        Running on Gemini's free tier. Will pause if rate limits are reached.
        """
    )

if not selected_col:
    st.warning("Please index at least one bank agreement into ChromaDB to begin.")
    st.stop()

collection = chroma_client.get_collection(name=selected_col)

# Display chat history
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "response_code" in message:
            code = message["response_code"]
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.caption(f" ")
            with col_b:
                current_vote = st.session_state.user_votes.get(code, None)
                default_idx = 0 if current_vote == "👍" else (1 if current_vote == "👎" else None)
                
                feedback = st.radio(
                    "Was this helpful?", 
                    ["👍", "👎"], 
                    key=f"fb_{code}", 
                    index=default_idx, 
                    label_visibility="collapsed", 
                    horizontal=True
                )
                if feedback and feedback != current_vote:
                    handle_vote(code, feedback)
                    st.rerun()

# --- 6. RAG QUERY EXECUTION & GOOGLE SHEETS LOGGING ---
if prompt := st.chat_input(f"Ask a question about {bank_cfg['display_name']}..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching document sections and generating response..."):
            try:
                n_results = 3
                temperature = 0
                llm_model = "gemini-3.5-flash-lite"
                embedding_model = "all-MiniLM-L6-v2"
                current_bank_name = bank_cfg['display_name']

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
                    
                    context_block += f"\n--- Institution: {current_bank_name} | Section: {title} (Page {page}) ---\n{doc}\n"
                    sources_cited.add(f"• **{title}** (Page {page})")
                    
                    chunks_referenced_audit.append({
                        "chunk_index": doc_idx + 1,
                        "bank": current_bank_name,
                        "section_title": title,
                        "page": page,
                        "toc_path": path,
                        "distance_score": float(dist),
                        "content_snippet": doc[:150] + "..."
                    })

                system_instruction = (
                    f"You are a precise, helpful general banking assistant. You are NOT representing any specific bank. "
                    "Answer the user's question strictly using the provided context excerpts from the banking agreement. "
                    "If the answer cannot be found in the context, state clearly that the document does not cover it. "
                    f"You must explicitly state that you are answering according to {current_bank_name}'s deposit agreement in your response, "
                    "and cite the relevant section title and page number."
                )

                user_prompt = (
                    f"Target Bank: {current_bank_name}\n"
                    f"Question: {prompt}\n\n"
                    f"Retrieved Document Context:\n{context_block}"
                )

                response = ai_client.models.generate_content(
                    model=llm_model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=temperature,
                        thinking_config=types.ThinkingConfig(thinking_level="minimal"),
                    ),
                )
                
                answer_text = response.text
                
                if sources_cited:
                    answer_text += "\n\n**Sources Referenced:**\n" + "\n".join(sources_cited)

                response_code = f"RAG-{uuid.uuid4().hex[:6].upper()}"
                answer_text += f"\n\n--- \n*Response ID Code: `{response_code}`*"

                audit_record = {
                    "response_code": response_code,
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "bank_institution": current_bank_name,
                    "collection_name": selected_col,
                    "user_query": prompt,
                    "llm_model": llm_model,
                    "embedding_model": embedding_model,
                    "parameters": {
                        "temperature": temperature,
                        "top_k_results": n_results
                    },
                    "assistant_response": answer_text,
                    "chunks_referenced": str(chunks_referenced_audit),
                    "sources_cited": str(list(sources_cited))
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
                            audit_record["llm_model"],
                            audit_record["embedding_model"],
                            str(audit_record["parameters"]),
                            audit_record["chunks_referenced"],
                            audit_record["assistant_response"],
                            0, # Initial Thumbs Up Count
                            0  # Initial Thumbs Down Count
                        ])
                    except Exception as sheet_err:
                        st.sidebar.warning(f"Could not write to Google Sheet: {sheet_err}")

                st.markdown(answer_text)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer_text,
                    "response_code": response_code
                })
                
                feedback = st.radio(
                    "Was this helpful?", 
                    ["👍", "👎"], 
                    key=f"fb_{response_code}", 
                    index=None, 
                    label_visibility="collapsed", 
                    horizontal=True
                )
                if feedback:
                    handle_vote(response_code, feedback)
                    st.rerun()

            except Exception as e:
                error_msg = f"**Usage Limit / Error Encountered:** `{str(e)}`\n\nThe free-tier usage limit has been reached or a network error occurred. Please wait a moment before trying again."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
