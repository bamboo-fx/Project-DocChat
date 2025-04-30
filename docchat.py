from groq import Groq
from dotenv import load_dotenv
import os
import pdfplumber
from bs4 import BeautifulSoup
import argparse
import requests
import sys

# Load environment variables from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

MODEL_NAME = "mixtral-8x7b-32768"  # You can change this to another Groq-supported model

def read_txt_file(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read()

def read_pdf_file(path):
    text = ''
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def read_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    return soup.get_text()

def load_text(filepath_or_url):
    if filepath_or_url.startswith("http"):
        return read_url(filepath_or_url)
    elif filepath_or_url.endswith(".pdf"):
        return read_pdf_file(filepath_or_url)
    else:
        return read_txt_file(filepath_or_url)

def chunk_text_by_words(text, chunk_size=200):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def summarize(prompt):
    print("Sending prompt to Groq...")
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error from Groq API: {e}")
        return "Groq API error."

def get_language(text):
    chunk = chunk_text_by_words(text, 100)[0]
    prompt = chunk + "\nWhat language is the above text in? Answer in one word."
    response = summarize(prompt)
    return response.strip().split()[0]

def main():
    parser = argparse.ArgumentParser(description="Chat with a document using Groq.")
    parser.add_argument("path", help="Path to a local .txt/.pdf file or a URL")
    args = parser.parse_args()

    try:
        text = load_text(args.path)
    except Exception as e:
        print(f"Could not load the file: {e}")
        sys.exit(1)

    language = get_language(text)
    print(f"\nDetected Language: {language}\n")

    chunks = chunk_text_by_words(text)
    print(f"Loaded {len(chunks)} chunks of text. Start chatting! (Type 'exit' to quit)\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        prompt = f"Based on the following document text:\n\n{text[:2000]}\n\nAnswer the question: {user_input}"
        response = summarize(prompt)
        print(f"DocChat: {response}\n")

if __name__ == "__main__":
    main()
