import os
import streamlit as st
import chromadb
from google import genai
from google.genai import types

# --- 1. CONFIGURATION & CLIENT SETUP ---
st.set_page_config(page_title="Multi-Bank RAG Assistant", layout="wide")

# Initialize Gemini Client (reads GEMINI_API_KEY from environment or Streamlit secrets)
# To keep costs at $0, use a free-tier API key from Google AI Studio.
api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
if not api_key:
    st.error("⚠️ GEMINI_API_KEY not found! Please set it in your environment variables or Streamlit secrets.")
    st.stop()

ai_client = genai.Client(api_key=api_key)

# Initialize ChromaDB persistent client
@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path="./chroma_db")

chroma_client = get_chroma_client()

# --- 2. SIDEBAR: BANK SELECTION ---
st.sidebar.title("🏦 Bank Selection")
st.sidebar.markdown("Choose a banking agreement to query:")

# Discover available bank collections in ChromaDB dynamically
try:
    collections = chroma_client.list_collections()
    bank_names = [col.name.replace("doc_", "").replace("_", " ").title() for col in collections]
    collection_map = {col.name.replace("doc_", "").replace("_", " ").title(): col.name for col in collections}
except Exception:
    bank_names = []
    collection_map = {}

if not bank_names:
    st.sidebar.warning("No bank documents found in ChromaDB yet! Run your chunking script first.")
    selected_bank_display = None
else:
    selected_bank_display = st.sidebar.selectbox("Select Institution:", bank_names)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Demo Safety Note:**\n"
    "Powered by the Gemini Free Tier. If request volumes exceed free-tier limits, "
    "the app will temporarily pause requests with zero financial charge to you."
)

# --- 3. MAIN INTERFACE ---
st.title("📄 Banking Agreement RAG Assistant")
if selected_bank_display:
    st.markdown(f"Currently querying: **{selected_bank_display}** agreement terms.")
else:
    st.warning("Please index at least one bank agreement into ChromaDB to begin.")
    st.stop()

# Connect to the specific bank's Chroma collection
active_collection_name = collection_map[selected_bank_display]
collection = chroma_client.get_collection(name=active_collection_name)

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. RAG QUERY EXECUTION ---
if prompt := st.chat_input(f"Ask a question about {selected_bank_display}..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching document sections and generating response..."):
            try:
                # Step A: Retrieve relevant context from ChromaDB (top 3 matching chunks)
                results = collection.query(
                    query_texts=[prompt],
                    n_results=3
                )
                
                retrieved_docs = results.get("documents", [[]])[0]
                retrieved_metas = results.get("metadatas", [[]])[0]
                
                # Format context package for the LLM
                context_block = ""
                sources_cited = set()
                
                for doc, meta in zip(retrieved_docs, retrieved_metas):
                    title = meta.get("section_title", "General Section")
                    page = meta.get("start_page", "Unknown")
                    context_block += f"\n--- Section: {title} (Page {page}) ---\n{doc}\n"
                    sources_cited.add(f"• **{title}** (Page {page})")

                # Step B: Construct prompt engineering instructions
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

                # Step C: Call Gemini (using gemini-2.5-flash for fast, zero-cost, high-efficiency responses)
                response = ai_client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.2,  # Low temperature for factual accuracy
                    ),
                )
                
                answer_text = response.text
                
                # Append citation references if available
                if sources_cited:
                    answer_text += "\n\n**Sources Referenced:**\n" + "\n".join(sources_cited)

                st.markdown(answer_text)
                st.session_state.messages.append({"role": "assistant", "content": answer_text})

            except Exception as e:
                # Handles free-tier rate limits (HTTP 429) or connection drops cleanly
                error_msg = f"⚠️ **Usage Limit / Error Encountered:** `{str(e)}`\n\nThe free-tier usage limit has been reached or a network error occurred. Please wait a moment before trying again."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})