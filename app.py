import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from langchain_community.document_loaders import CSVLoader, TextLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATA_DIR = "data"
VECTOR_DIR = "vectorstore"


st.set_page_config(
    page_title="AI Financial Operations Copilot",
    page_icon="💼",
    layout="wide"
)

st.title("💼 AI Financial Operations Copilot")
st.caption("GenAI + RAG assistant for CSV financial data and TXT policy rules")


def get_groq_client():
    if not GROQ_API_KEY:
        st.error("Missing GROQ_API_KEY. Add it inside your .env file.")
        st.stop()
    return Groq(api_key=GROQ_API_KEY)


def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)


def get_csv_files():
    ensure_data_dir()
    return [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")]


def get_txt_files():
    ensure_data_dir()
    return [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".txt")]


def load_selected_csv(selected_csv):
    if not selected_csv:
        return None

    path = os.path.join(DATA_DIR, selected_csv)

    if not os.path.exists(path):
        return None

    return pd.read_csv(path)


def load_selected_policy_text(selected_policy):
    if not selected_policy:
        return "No policy file selected."

    path = os.path.join(DATA_DIR, selected_policy)

    if not os.path.exists(path):
        return "Selected policy file not found."

    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return "Selected policy file is empty."

    return content


def load_documents_for_rag():
    docs = []
    ensure_data_dir()

    for file in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, file)

        if file.lower().endswith(".csv"):
            loader = CSVLoader(file_path=file_path)
            docs.extend(loader.load())

    txt_loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    docs.extend(txt_loader.load())

    return docs


def build_rag_index():
    docs = load_documents_for_rag()

    if not docs:
        st.warning("No CSV or TXT files found. Upload files first.")
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=300
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DIR
    )

    vectorstore.persist()
    return vectorstore


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return Chroma(
        persist_directory=VECTOR_DIR,
        embedding_function=embeddings
    )


def retrieve_context(question, selected_csv=None, selected_policy=None, k=8):
    vectorstore = load_vectorstore()

    docs = vectorstore.similarity_search(question, k=30)
    filtered_docs = []

    for doc in docs:
        source = str(doc.metadata.get("source", "")).lower()

        csv_match = selected_csv and selected_csv.lower() in source
        policy_match = selected_policy and selected_policy.lower() in source

        if csv_match or policy_match:
            filtered_docs.append(doc)

    if filtered_docs:
        docs = filtered_docs[:k]
    else:
        docs = docs[:k]

    context = "\n\n".join([doc.page_content for doc in docs])

    return context, docs


def ask_finance_ai(question, df, selected_csv, selected_policy, policy_text, retrieved_context):
    client = get_groq_client()

    csv_table_context = "No CSV selected."

    if df is not None:
        max_rows = 100
        limited_df = df.head(max_rows)

        csv_table_context = f"""
Selected CSV file: {selected_csv}
Rows shown to you: {len(limited_df)} out of {len(df)}
Columns: {", ".join(df.columns)}

CSV Table Preview:
{limited_df.to_markdown(index=False)}
"""

    prompt = f"""
You are an AI Financial Operations Copilot.

You analyze financial CSV data and company policy TXT files.

Use ONLY the provided context.
Do not invent invoice IDs, vendors, dates, amounts, departments, PO numbers, statuses, risk categories, or policy rules.

If the question is about invoice/payment data, use the CSV table context and retrieved CSV chunks.
If the question is about company rules/policy, use the Selected Policy Text first.
If both are relevant, combine CSV facts with policy rules.

Selected CSV:
{selected_csv}

Selected Policy TXT:
{selected_policy}

Selected Policy Text:
{policy_text}

CSV Table Context:
{csv_table_context}

Retrieved RAG Context:
{retrieved_context}

User Question:
{question}

Answer clearly and practically.
If listing records, include important fields such as invoice ID, vendor/supplier, amount, paid amount, status, due date, department/team, PO, or risk category when available.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a careful financial operations AI assistant. Be accurate and do not hallucinate."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1
    )

    return response.choices[0].message.content


def validate_invoice_agent(invoice_text, df, selected_csv, selected_policy, policy_text, retrieved_context):
    client = get_groq_client()

    csv_table_context = "No CSV selected."

    if df is not None:
        max_rows = 100
        limited_df = df.head(max_rows)

        csv_table_context = f"""
Selected CSV file: {selected_csv}
Rows shown to you: {len(limited_df)} out of {len(df)}
Columns: {", ".join(df.columns)}

CSV Table Preview:
{limited_df.to_markdown(index=False)}
"""

    prompt = f"""
You are an AI invoice validation agent.

Use the selected CSV context, selected policy text, and retrieved RAG context to validate the pasted invoice.

Do not invent policy rules.
Do not invent invoice data.
If the invoice is not present in the selected CSV, say that clearly and then analyze only the pasted invoice text using the selected policy.

Look for:
- missing PO number
- unpaid or pending payment status
- payment amount mismatch
- unusually high invoice amount
- high risk category
- due date concerns
- vendor or department risk
- policy violations

Selected CSV:
{selected_csv}

Selected Policy TXT:
{selected_policy}

Selected Policy Text:
{policy_text}

CSV Table Context:
{csv_table_context}

Retrieved RAG Context:
{retrieved_context}

Invoice Text:
{invoice_text}

Return the result in this format:

Risk Level:
Issues Found:
Recommended Action:
Explanation:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You validate invoices using financial CSV context and selected company policy context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1
    )

    return response.choices[0].message.content


with st.sidebar:
    st.header("📄 File Upload")

    uploaded_files = st.file_uploader(
        "Upload CSV and TXT files",
        type=["csv", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        ensure_data_dir()

        for uploaded_file in uploaded_files:
            file_path = os.path.join(DATA_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        st.success("File(s) uploaded successfully.")

    if st.button("Build / Refresh RAG Index"):
        with st.spinner("Building RAG index with LangChain + ChromaDB + Hugging Face embeddings..."):
            build_rag_index()
        st.success("RAG index created successfully.")

    st.divider()

    st.header("📊 CSV Selection")
    csv_files = get_csv_files()

    selected_csv = None
    selected_df = None

    if csv_files:
        selected_csv = st.selectbox("Select CSV for invoice/payment data", csv_files)
        selected_df = load_selected_csv(selected_csv)

        if selected_df is not None:
            st.caption(
                f"Selected CSV has {len(selected_df)} rows and {len(selected_df.columns)} columns."
            )
    else:
        st.info("Upload a CSV to enable invoice/payment analysis.")

    st.divider()

    st.header("📜 Policy Selection")
    txt_files = get_txt_files()

    selected_policy = None
    policy_text = "No policy file selected."

    if txt_files:
        selected_policy = st.selectbox("Select TXT policy file", txt_files)
        policy_text = load_selected_policy_text(selected_policy)

        with st.expander("Preview selected policy"):
            st.write(policy_text)
    else:
        st.info("Upload a TXT policy file to enable policy-based analysis.")

    st.divider()

    st.write("Try these questions:")
    st.code("Summarize policy.")
    st.code("Which invoices violate company policy?")
    st.code("Which invoices require director approval?")
    st.code("Which invoices have missing PO numbers?")
    st.code("List all high risk invoices.")
    st.code("Compare BILL-1001 with BILL-1013.")


tab1, tab2, tab3 = st.tabs(
    [
        "💬 AI Finance Chat",
        "🤖 Invoice Validation Agent",
        "📊 CSV Preview"
    ]
)


with tab1:
    st.subheader("AI Finance Chat")
    st.write(
        "Ask natural-language questions about selected CSV invoice/payment data and selected TXT policy rules."
    )

    question = st.text_input("Ask a finance question")

    if st.button("Ask AI"):
        if not question:
            st.warning("Please enter a question.")
        elif selected_df is None:
            st.warning("Please upload and select a CSV first.")
        else:
            with st.spinner("Analyzing CSV and policy context with GenAI + RAG..."):
                try:
                    retrieved_context, docs = retrieve_context(
                        question,
                        selected_csv=selected_csv,
                        selected_policy=selected_policy,
                        k=8
                    )

                    answer = ask_finance_ai(
                        question=question,
                        df=selected_df,
                        selected_csv=selected_csv,
                        selected_policy=selected_policy,
                        policy_text=policy_text,
                        retrieved_context=retrieved_context
                    )

                    st.markdown("### Answer")
                    st.write(answer)

                except Exception as e:
                    st.error(f"Error: {e}")
                    st.info("Click 'Build / Refresh RAG Index' first.")


with tab2:
    st.subheader("Agentic AI: Invoice Validation")
    st.write("Paste an invoice and let AI review it using selected CSV + selected TXT policy context.")

    invoice_input = st.text_area(
        "Paste invoice text here",
        height=260,
        value="""Invoice Number: BILL-1014
Vendor: CyberShield LLC
PO Number: PO-A114
Invoice Amount: 21500
Payment Status: Mismatch
Payment Amount: 20000
Invoice Date: 2026-04-28
Due Date: 2026-05-28
Department: Security
Risk Category: High"""
    )

    if st.button("Validate Invoice"):
        if selected_df is None:
            st.warning("Please upload and select a CSV first.")
        else:
            with st.spinner("Invoice validation agent is reviewing..."):
                try:
                    agent_query = invoice_input + "\n\ncompany policy invoice validation payment rules"
                    retrieved_context, docs = retrieve_context(
                        agent_query,
                        selected_csv=selected_csv,
                        selected_policy=selected_policy,
                        k=8
                    )

                    result = validate_invoice_agent(
                        invoice_text=invoice_input,
                        df=selected_df,
                        selected_csv=selected_csv,
                        selected_policy=selected_policy,
                        policy_text=policy_text,
                        retrieved_context=retrieved_context
                    )

                    st.markdown("### Agent Result")
                    st.write(result)

                except Exception as e:
                    st.error(f"Error: {e}")
                    st.info("Click 'Build / Refresh RAG Index' first.")


with tab3:
    st.subheader("CSV Preview")

    if selected_df is not None:
        st.write(f"Showing file: `{selected_csv}`")
        st.dataframe(selected_df, use_container_width=True)

        st.markdown("### CSV Summary")
        st.write(f"Rows: {len(selected_df)}")
        st.write(f"Columns: {len(selected_df.columns)}")
        st.write("Columns:")
        st.write(", ".join(selected_df.columns))
    else:
        st.warning("No CSV selected. Upload a CSV from the sidebar.")


st.divider()
st.caption(
    "Built with Streamlit, LangChain, ChromaDB, Hugging Face embeddings, Groq API, pandas, CSV RAG, and direct TXT policy retrieval."
)