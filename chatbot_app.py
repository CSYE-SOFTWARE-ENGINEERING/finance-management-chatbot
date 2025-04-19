from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st



from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough

# --- Load API key from environment
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("❌ OPENAI_API_KEY not set. Please check your environment or .env file.")
    st.stop()

# --- Vectorstore setup
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
persist_directory = "chroma_db"
# vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding)
docs = [Document(page_content="Peter Lynch was a legendary investor.")]
vectorstore = FAISS.from_documents(docs, embedding)

# --- Prompt Template & Chat Model
prompt_template = PromptTemplate.from_template(
    "You are a helpful financial assistant. Use the following context to answer the user's question.\n\n"
    "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
)
chat = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=openai_api_key)

# --- RAG Chain
def get_answer(question: str) -> str:
    retriever = vectorstore.as_retriever(
        search_type='mmr',
        search_kwargs={'k': 5, 'lambda_mult': 0.7}
    )
    chain = (
        {'context': retriever, 'question': RunnablePassthrough()}
        | prompt_template
        | chat
        | StrOutputParser()
    )
    return chain.invoke(question)

# --- Streamlit UI ---
st.set_page_config(page_title="Money-Bot 💬", page_icon="📈")
st.title("📊 Finance Management-Bot")
st.write("Ask me about how to manage your finances?")

user_question = st.text_input("Enter your question:")

if user_question:
    with st.spinner("Thinking..."):
        try:
            response = get_answer(user_question)
            st.success("Answer:")
            st.write(response)
        except Exception as e:
            st.error(f"❌ Failed to get answer: {e}")
