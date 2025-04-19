from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_openai.embeddings import OpenAIEmbeddings
import os

# For backtesting
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --- Load API key from environment
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("❌ OPENAI_API_KEY not set. Please check your environment or .env file.")
    st.stop()

# --- Vectorstore setup
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
persist_directory = "chroma_db"
vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding)

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

# ----------------------------
# --- Streamlit Chatbot UI ---
# ----------------------------
st.set_page_config(page_title="Peter-Bot 💬", page_icon="📈")
st.title("📊 Plynch-Bot")
st.write("Ask anything about Peter Lynch’s strategies, trading psychology, or risk.")

user_question = st.text_input("Enter your question:")

if user_question:
    with st.spinner("Thinking..."):
        try:
            response = get_answer(user_question)
            st.success("Answer:")
            st.write(response)
        except Exception as e:
            st.error(f"❌ Failed to get answer: {e}")

# ----------------------------
# --- Strategy Backtest UI ---
# ----------------------------
st.divider()
st.header("📈 Stock Strategy Backtester")

stock_ticker = st.text_input("Enter stock symbol (e.g. AAPL, TSLA):", value="AAPL")
start_date = st.date_input("Start date", pd.to_datetime("2023-01-01"))
end_date = st.date_input("End date", pd.to_datetime("2024-01-01"))

strategy = st.selectbox("Select Strategy", ["Moving Average Crossover", "RSI", "MACD"])
short_window = st.slider("Short Moving Average (days)", 5, 50, 10) if strategy == "Moving Average Crossover" else None
long_window = st.slider("Long Moving Average (days)", 20, 200, 50) if strategy == "Moving Average Crossover" else None

if st.button("Run Backtest"):
    df = yf.download(stock_ticker, start=start_date, end=end_date)

    if df.empty:
        st.error("No data found for this symbol and date range.")
    else:
        fig, ax = plt.subplots(figsize=(10, 5))

        if strategy == "Moving Average Crossover":
            df["SMA_Short"] = df["Close"].rolling(window=short_window).mean()
            df["SMA_Long"] = df["Close"].rolling(window=long_window).mean()
            df["Signal"] = 0
            df.loc[df.index[short_window:], "Signal"] = np.where(
                df["SMA_Short"].iloc[short_window:] > df["SMA_Long"].iloc[short_window:], 1, 0
            )
            df["Position"] = df["Signal"].diff()

            ax.plot(df["Close"], label="Close Price", alpha=0.5)
            ax.plot(df["SMA_Short"], label=f"SMA {short_window}", color="green")
            ax.plot(df["SMA_Long"], label=f"SMA {long_window}", color="red")
            ax.plot(df[df["Position"] == 1].index, df["Close"][df["Position"] == 1], "^", markersize=10, color="g", label="Buy Signal")
            ax.plot(df[df["Position"] == -1].index, df["Close"][df["Position"] == -1], "v", markersize=10, color="r", label="Sell Signal")

        elif strategy == "RSI":
            delta = df["Close"].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss
            df["RSI"] = 100 - (100 / (1 + rs))
            ax.plot(df["RSI"], label="RSI", color="purple")
            ax.axhline(70, color='red', linestyle='--')
            ax.axhline(30, color='green', linestyle='--')
            ax.set_title("RSI (Relative Strength Index)")

        elif strategy == "MACD":
            short_ema = df["Close"].ewm(span=12, adjust=False).mean()
            long_ema = df["Close"].ewm(span=26, adjust=False).mean()
            df["MACD"] = short_ema - long_ema
            df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
            ax.plot(df["MACD"], label="MACD", color="blue")
            ax.plot(df["Signal_Line"], label="Signal Line", color="orange")
            ax.axhline(0, linestyle="--", color="gray")
            ax.set_title("MACD (Moving Average Convergence Divergence)")

        ax.legend()
        st.pyplot(fig)

        # Export CSV
        st.session_state["backtest_df"] = df
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Backtest Data as CSV", csv, f"{stock_ticker}_backtest.csv", "text/csv")

# ----------------------------
# --- Fundamentals Panel ---
# ----------------------------
st.divider()
st.subheader("💡 Basic Fundamentals")

try:
    info = yf.Ticker(stock_ticker).info
    st.markdown(f"""
    - **Company**: {info.get("longName", "N/A")}
    - **Sector**: {info.get("sector", "N/A")}
    - **Industry**: {info.get("industry", "N/A")}
    - **Market Cap**: {info.get("marketCap", "N/A")}
    - **P/E Ratio**: {info.get("trailingPE", "N/A")}
    - **52-Week High**: {info.get("fiftyTwoWeekHigh", "N/A")}
    - **52-Week Low**: {info.get("fiftyTwoWeekLow", "N/A")}
    - **Volume**: {info.get("volume", "N/A")}
    """)
except Exception as e:
    st.error(f"❌ Could not load fundamentals: {e}")







