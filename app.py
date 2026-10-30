import os
import uuid
import datetime
import streamlit as st
import chromadb
from google import genai
from google.genai import types

# --- 1. CONFIGURATION & CLIENT SETUP ---
st.set_page_config(page_title="Multi-Bank RAG Assistant", layout="wide")

# Initialize Gemini Client
api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
if not api_key:
    st.error("Error: No API KEY")
    st.stop()

ai_client = genai.Client(api_key=api_key)

# Initialize ChromaDB persistent client
@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path="./chroma_db")

chroma_client = get_chroma_client()

# --- 2. MAPPINGS & CONFIGS ---
# Map raw collection names / identifiers to clean display names and color themes
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

# Fallback helper for any collection names not explicitly in the map
def get_bank_config(col_name):
    clean_key = col_name.replace("doc_", "").lower()
    for key, cfg in BANK_CONFIGS.items():
        if key in clean_key:
            return cfg
    # Default fallback formatting
    formatted = col_name.replace("doc_", "").replace("_", " ").title()
    return {"display_name": formatted, "primary": "#1f77b4", "accent": "#ff7f0e"}

# --- 3. SIDEBAR: BANK SELECTION & LOOKUP ---
st.sidebar.title("🏦 Bank Selection")

try:
    collections = chroma_client.list_collections()
    collection_list = [col.name for col in collections]
except Exception:
    collection_list = []

if not collection_list:
    st.sidebar.warning("No docs found in ChromaDB, run chunking")
    selected_col = None
    bank_cfg = {"display_name": "None", "primary": "#111111", "accent": "#888888"}
else:
    # Build selectbox mapping collection names to display names
    col_display_map = {col: get_bank_config(col)["display_name"] for col in collection_list}
    selected_col = st.sidebar.selectbox("Select Institution:", list(col_display_map.keys()), format_func=lambda x: col_display_map[x])
    bank_cfg = get_bank_config(selected_col)

# --- APPLY DYNAMIC THEME & BOTTOM UNDERLINE ---
st.markdown(
    f"""
    <style>
        /* Main background subtle accent tint */
        .stApp {{
            background: linear-gradient(to bottom, {bank_cfg['primary']}0D, #FFFFFF 20%);
        }}
        /* Bottom accent underline banner */
        .bottom-underline {{
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 6px;
            background-color: {bank_cfg['accent']};
            z-index: 99999;
        }}
        /* Sidebar styling accent */
        [data-testid="stSidebar"] {{
            border-right: 2px solid {bank_cfg['primary']}33;
        }}
    </style>
    <div class="bottom-underline"></div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

# --- UNIQUE RESPONSE CODE LOOKUP SECTION ---
st.sidebar.subheader("🔍 Metadata Inspector")
lookup_code = st.sidebar.text_input("Enter Response Code:", placeholder="e.g., RAG-A1B2C3")
if lookup_code:
    if "response_metadata_store" in st.session_state and lookup_code.strip() in st.session_state.response_metadata_store:
        st.sidebar.success("Found matching response audit record!")
        meta_record = st.session_state.response_metadata_store[lookup_code.strip()]
        with st.sidebar.expander("View Audit Details", expanded=True):
            st.json(meta_record)
    else:
        st.sidebar.error("Code not found in session history.")

st.sidebar.markdown("---")
st.sidebar.info(
    "**Demo Safety Note:**\n"
    "Powered by the Gemini Free Tier. If request volumes exceed limits, "
    "requests will pause with zero financial charge."
)

# --- 4. MAIN INTERFACE ---
st.title(f"📄 {bank_cfg['display_name']} Agreement Assistant")
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
        # If assistant message has an associated metadata code, display it & feedback
        if message["role"] == "assistant" and "response_code" in message:
            code = message["response_code"]
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.caption(f"🔑 Response Code: `{code}` (Paste in sidebar to inspect metadata)")
            with col_b:
                feedback = st.radio("Was this helpful?", ["👍 Yes", "👎 No", "None"], key=f"fb_{code}", label_visibility="collapsed", horizontal=True)

# --- 5. RAG QUERY EXECUTION & METADATA TRACKING ---
if prompt := st.chat_input(f"Ask a question about {bank_cfg['display_name']}..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching document sections and generating response..."):
            try:
                # Parameters config
                n_results = 3
                temperature = 0.2
                llm_model = "gemini-3.8-flash"
                embedding_model = "Chroma Default (all-MiniLM-L6-v2)"

                # Step A: Retrieve relevant context from ChromaDB
                results = collection.query(
                    query_texts=[prompt],
                    n_results=n_results,
                    include=["documents", "metadatas", "distances"]
                )
                
                retrieved_docs = results.get("documents", [[]])[0]
                retrieved_metas = results.get("metadatas", [[]])[0]
                retrieved_distances = results.get("distances", [[]])[0] if "distances" in results else [0.0] * len(retrieved_docs)
                
                # Format context package and compile chunk metadata details
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

                # Step B: Construct prompt instructions
                system_instruction = (
                    "You are a precise, helpful banking assistant. Answer the user's question "
                    "strictly using the provided context excerpts from the banking agreement. "
                    "If the answer cannot be found in the context, state clearly that the document "
                    "does not cover it. Always cite the relevant section title and page number."
                )

                user_prompt = (
                    f"Question: {prompt}\n\n"
                    f"Retrieved Document Context:\n{context_block}"
                )

                # Step C: Call Gemini
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

                # Generate unique response code
                response_code = f"RAG-{uuid.uuid4().hex[:6].upper()}"
                answer_text += f"\n\n--- \n🔑 *Response ID Code: `{response_code}`*"

                # Save comprehensive metadata record
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

                st.markdown(answer_text)
                
                # Append to messages with response code attached
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer_text,
                    "response_code": response_code
                })
                
                # Show feedback widget immediately for new message
                st.radio("Was this helpful?", ["👍 Yes", "👎 No", "None"], key=f"fb_{response_code}", label_visibility="collapsed", horizontal=True)

            except Exception as e:
                error_msg = f"**Usage Limit / Error Encountered:** `{str(e)}`\n\nThe free-tier usage limit has been reached or a network error occurred. Please wait a moment before trying again."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
