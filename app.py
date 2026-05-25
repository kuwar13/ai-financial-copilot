import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from langchain_community.document_loaders import DirectoryLoader, TextLoader, CSVLoader, PyPDFLoader
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
st.caption("GenAI + RAG assistant for financial documents, invoices, policies, and CSV analysis")


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


def load_selected_csv(selected_csv):
    if not selected_csv:
        return None

    path = os.path.join(DATA_DIR, selected_csv)

    if not os.path.exists(path):
        return None

    return pd.read_csv(path)


def load_documents_langchain():
    docs = []
    ensure_data_dir()

    txt_loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    docs.extend(txt_loader.load())

    for file in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, file)

        if file.lower().endswith(".csv"):
            docs.extend(CSVLoader(file_path=file_path).load())

        if file.lower().endswith(".pdf"):
            docs.extend(PyPDFLoader(file_path).load())

    return docs


def build_langchain_vectorstore():
    docs = load_documents_langchain()

    if not docs:
        st.warning("No documents found in data folder.")
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


def load_langchain_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return Chroma(
        persist_directory=VECTOR_DIR,
        embedding_function=embeddings
    )


def retrieve_context(query, k=8):
    vectorstore = load_langchain_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return context, docs


def ask_general_finance_ai(question, df, selected_csv, retrieved_context):
    client = get_groq_client()

    csv_context = "No CSV selected."

    if df is not None:
        max_rows = 100
        limited_df = df.head(max_rows)
        csv_context = f"""
Selected CSV file: {selected_csv}
Rows shown to you: {len(limited_df)} out of {len(df)}
Columns: {", ".join(df.columns)}

CSV Table:
{limited_df.to_markdown(index=False)}
"""

    prompt = f"""
You are an AI Financial Operations Copilot.

You may receive two types of context:
1. CSV table context for invoice/payment records.
2. Retrieved document context from uploaded PDFs/TXT/CSV files, such as policies, invoice text, audit rules, and payment guidelines.

Use the available context to answer the user's question.

Rules:
- Use ONLY the provided context.
- Do not invent invoice IDs, vendors, dates, amounts, departments, or policy rules.
- If the question is about the CSV, analyze the CSV table directly.
- If the question is about policy or document rules, use the retrieved document context.
- If both are relevant, combine CSV facts with policy context.
- If the answer is not available, say you do not have enough information.

CSV Context:
{csv_context}

Retrieved Document Context:
{retrieved_context}

User Question:
{question}

Answer clearly and practically.
If listing records, include important fields such as invoice ID, vendor/supplier, amount, status, due date, department/team, PO, or risk category when available.
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


def validate_invoice_agent(invoice_text):
    policy_query = "company payment policy invoice approval purchase order PO number payment mismatch overdue invoice rules"
    policy_context, policy_docs = retrieve_context(policy_query, k=8)

    client = get_groq_client()

    prompt = f"""
You are an AI invoice validation agent.

Use the uploaded policy/document context below to validate the invoice.
Do NOT invent policy rules.
If the policy is missing, say policy context is missing.

Policy / Document Context:
{policy_context}

Invoice:
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
                "content": "You validate invoices using uploaded company policy context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1
    )

    return response.choices[0].message.content, policy_docs


with st.sidebar:
    st.header("📄 Document Controls")

    uploaded_files = st.file_uploader(
        "Upload TXT, PDF, or CSV files",
        type=["txt", "pdf", "csv"],
        accept_multiple_files=True
    )

    if uploaded_files:
        ensure_data_dir()

        for uploaded_file in uploaded_files:
            file_path = os.path.join(DATA_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        st.success("Files uploaded successfully.")

    if st.button("Build / Refresh RAG Index"):
        with st.spinner("Building LangChain + ChromaDB vector index..."):
            build_langchain_vectorstore()
        st.success("RAG index created successfully.")

    st.divider()

    st.header("📊 CSV Selection")
    csv_files = get_csv_files()

    selected_csv = None
    selected_df = None

    if csv_files:
        selected_csv = st.selectbox("Select CSV for analysis", csv_files)
        selected_df = load_selected_csv(selected_csv)

        if selected_df is not None:
            st.caption(f"Selected CSV has {len(selected_df)} rows and {len(selected_df.columns)} columns.")
    else:
        st.info("Upload a CSV to enable CSV analysis.")

    st.divider()

    st.write("Try these questions:")
    st.code("How many records are there?")
    st.code("List all high risk invoices.")
    st.code("Compare BILL-1001 with BILL-1013.")
    st.code("Which invoices are due before 2026-05-01?")
    st.code("Does BILL-1013 violate company policy?")
    st.code("Summarize the payment policy.")


tab1, tab2, tab3 = st.tabs(
    [
        "💬 AI Finance Chat",
        "🤖 Policy-Based Invoice Agent",
        "📊 CSV Preview"
    ]
)


with tab1:
    st.subheader("AI Finance Chat")
    st.write("Ask questions about uploaded CSVs, invoices, policies, PDFs, and payment documents.")

    question = st.text_input("Ask a finance question")

    if st.button("Ask AI"):
        if not question:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Analyzing uploaded data and documents..."):
                try:
                    retrieved_context, docs = retrieve_context(question, k=8)
                    answer = ask_general_finance_ai(
                        question=question,
                        df=selected_df,
                        selected_csv=selected_csv,
                        retrieved_context=retrieved_context
                    )

                    st.markdown("### Answer")
                    st.write(answer)

                    # st.markdown("### Retrieved Sources")
                    # for i, doc in enumerate(docs, start=1):
                    #     source = doc.metadata.get("source", "Unknown source")
                    #     st.write(f"**Source {i}:** {source}")
                    #     st.info(doc.page_content[:700])

                except Exception as e:
                    st.error(f"Error: {e}")
                    st.info("Click 'Build / Refresh RAG Index' first.")


with tab2:
    st.subheader("Agentic AI: Policy-Based Invoice Validation")
    st.write("This agent retrieves uploaded policy context first, then validates invoice text.")

    invoice_input = st.text_area(
        "Paste invoice text here",
        height=260,
        value="""Invoice Number: INV-002
Vendor: DataSecure Inc
PO Number: Missing
Invoice Amount: 12500
Payment Status: Unpaid
Payment Amount: 0
Invoice Date: 2026-04-01
Due Date: 2026-05-01
Department: Finance"""
    )

    if st.button("Validate Invoice Against Uploaded Policy"):
        with st.spinner("Retrieving policy and validating invoice..."):
            try:
                result, policy_docs = validate_invoice_agent(invoice_input)

                st.markdown("### Agent Result")
                st.write(result)

                # st.markdown("### Policy Sources Used")
                # for i, doc in enumerate(policy_docs, start=1):
                #     source = doc.metadata.get("source", "Unknown source")
                #     st.write(f"**Source {i}:** {source}")
                #     st.info(doc.page_content[:700])

            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Make sure you uploaded a policy TXT/PDF and clicked Build / Refresh RAG Index.")


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
        st.warning("No CSV selected. Upload a CSV from Document Controls.")


st.divider()
st.caption(
    "Built with Streamlit, LangChain, ChromaDB, Hugging Face embeddings, Groq API, and direct LLM-based CSV analysis."
)