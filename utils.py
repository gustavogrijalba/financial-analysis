import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
from dotenv import load_dotenv

#load groq
load_dotenv()
client = Groq(
    api_key= os.getenv("GROQ_API_KEY"),
)

def parse_article(url):
    """Parse the article from the given URL and return the text."""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        article_text = ' '.join([para.get_text() for para in paragraphs])
        return article_text
    except Exception as e:
        raise Exception(f"Error parsing the article: {e}")

def get_stock_tickers_from_article(article_text):
    prompt = f"""You are an expert financial analyst with a deep understanding of the stock market.
    Your task is to analyze the following article and extract any stock tickers (e.g., AAPL for Apple, TSLA for Tesla) mentioned or strongly implied based on company names or context. Focus only on publicly traded companies.

    <article_text>
    {article_text}
    </article_text>

    Please return the stock tickers in JSON format, where each ticker is provided along with a specific explanation in the following structure:
    {{
        "tickers": [
            {{"ticker": "AAPL", "explanation": "Apple is mentioned in the article for its recent financial performance."}},
            {{"ticker": "TSLA", "explanation": "Tesla is referenced in the context of electric vehicles."}},
            {{"ticker": "MSFT", "explanation": "Microsoft is noted for its advancements in AI."}}
        ]
    }}

    Additional considerations:
    1. If a company name is mentioned, map it to its ticker symbol if it is publicly traded.
    2. Include only unique tickers. Avoid duplicates.
    3. If no relevant tickers are found, return an empty list like this: {{ "tickers": [] }}
    4. Use reliable mapping of company names to tickers; do not make assumptions.

    Provide the result strictly in JSON format with no additional text."""
    
    llm_response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ],
        response_format = {"type" : "json_object"}
    )
    response = llm_response.choices[0].message.content
    return response

def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
    """
    Generates embeddings for the given text using a specified Hugging Face model.

    Args:
        text (str): The input text to generate embeddings for.
        model_name (str): The name of the Hugging Face model to use.
                          Defaults to "sentence-transformers/all-mpnet-base-v2".

    Returns:
        np.ndarray: The generated embeddings as a NumPy array.
    """
    model = SentenceTransformer(model_name)
    return model.encode(text)

