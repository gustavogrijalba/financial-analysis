import streamlit as st
import yfinance as yf
import utils as ut
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
import pandas as pd
from pinecone import Pinecone
from groq import Groq

#setup pinecone and environment
load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index("stocks")
namespace = "stock-descriptions"

#setup groq
client = Groq(
    api_key= os.getenv("GROQ_API_KEY"),
)

#similarity search with our pinecone db
def pinecone_query(query):
    system_prompt = f"""You are an expert at providing answers about stocks. Please answer my question provided.
"""
    raw_query_embedding = ut.get_huggingface_embeddings(query)
    top_matches = pinecone_index.query(vector=raw_query_embedding.tolist(), top_k=10, include_metadata=True, namespace=namespace)
    contexts = [item['metadata']['text'] for item in top_matches['matches']]
    augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[ : 10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query
    chat_completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_query}
        ]
    )
    response = chat_completion.choices[0].message.content
    return response

# streamlit ui
st.title("Stock Ticker Finder")
st.subheader("Chat with the assistant and find relevant stock tickers!")

#user input we will use for our query
article_url = st.text_input("Enter the article URL (optional):")
user_query = st.text_input("Enter your query:")

#article input
article_text = ""
#article url is optional, check if we can parse article url from input
if article_url:
    article_text = ut.parse_article(article_url)
    
    # display an excerpt of the article for more UI context
    article_excerpt = ' '.join(article_text.split()[:150]) + "..."
    
    #two column layout for tickers and excerpt
    col1, col2 = st.columns(2)

    # article content will be on the left
    with col1:
        st.subheader('Article Excerpt')
        st.write(article_excerpt)

    #tickers will be on the right
    with col2:
        try:
            #use utils.py to get the stock tickers from the article 
            tickers_with_explanation = ut.get_stock_tickers_from_article(article_text)
            tickers_data = eval(tickers_with_explanation)  #conver to dictionary
            tickers = tickers_data.get('tickers', [])

            if tickers:
                #display the tickers
                st.subheader("Relevant Stock Tickers:")
                for ticker_info in tickers:
                    st.write(f"**Ticker**: {ticker_info['ticker']}")
                    st.write(f"**Explanation**: {ticker_info['explanation']}")
            else:
                st.write("No relevant stock tickers found in the article.")
        except Exception as e:
            st.write(f"Error: {str(e)}")

    #create a stock price chart using yahoo finance
    st.subheader("Stock Price Chart (Last Year):")
    tickers_list = [ticker_info['ticker'] for ticker_info in tickers]

    #stock data
    if tickers_list:
        data = yf.download(tickers_list, period="1y", group_by="ticker")

        #plot data using plotly (i used gpt for this)
        fig, ax = plt.subplots(figsize=(10, 6))
        for ticker in tickers_list:
            data[ticker]['Close'].plot(ax=ax, label=ticker)

        ax.set_title("Stock Prices Over the Last Year")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        st.pyplot(fig)

if user_query:
    st.subheader("Relevant Stock Tickers for Your Query:")
    try:
        #call pinecone query and write it using streamlit to the uI
        gen_tickers = pinecone_query(user_query)
        st.write(gen_tickers)
    except Exception as e:
        st.write(f"Error: {str(e)}")