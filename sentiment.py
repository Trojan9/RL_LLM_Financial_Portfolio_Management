import os
import json
import pandas as pd
import requests

API_URL = "http://172.187.217.98:8000/v1/completions"
HEADERS = {"Content-Type": "application/json"}

def analyze_news(title):
    """
    Query the API to analyze the sentiment of a financial news headline.
    """
    prompt = f"Analyze the sentiment of this financial news headline and suggest actions (Buy, Hold, Sell): '{title}'"
    payload = {
        "model": "meta-llama/Llama-3.2-1B",
        "prompt": prompt,
        "max_tokens": 50,
        "temperature": 0.7
    }
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        result = response.json()
        sentiment = result.get("choices", [{}])[0].get("text", "").strip()
        return sentiment
    except requests.exceptions.RequestException as e:
        print(f"API error for headline: {title}\nError: {e}")
        return "Unknown"

def get_sentiments(news_csv_path, sentiments_path="sentiments.json"):
    """
    Analyze news titles from a CSV file and save their sentiments into a JSON file.

    - If `sentiments.json` exists, continue from where it left off.
    - Process the first 10% of the dataset after removing null or empty values.
    """
    # Load news data
    print("Loading news data...")
    news_data = pd.read_csv(news_csv_path)
    news_data = news_data.dropna(subset=["TITLE"])
    news_titles = news_data["TITLE"].tolist()
    total_titles = int(len(news_titles) * 0.1)  # First 10% of the dataset
    news_titles = news_titles[:total_titles]
    print(f"Total titles to process: {len(news_titles)}")

    # Load existing sentiments if the file exists
    if os.path.exists(sentiments_path):
        with open(sentiments_path, "r") as f:
            sentiments = json.load(f)
    else:
        sentiments = {}

    # Continue processing only missing titles
    start_index = len(sentiments)
    print(f"Resuming from index: {start_index}")
    for i in range(start_index, len(news_titles)):
        title = news_titles[i]
        print(f"Processing title {i + 1}/{len(news_titles)}: {title}")
        sentiment = analyze_news(title)
        sentiments[title] = sentiment

        # Save progress after every 10 titles
        if (i + 1) % 10 == 0 or (i + 1) == len(news_titles):
            with open(sentiments_path, "w") as f:
                json.dump(sentiments, f, indent=4)
            print(f"Progress saved: {len(sentiments)} sentiments stored.")

    print("Sentiment analysis completed.")
    return sentiments




# Set the file paths
news_csv_path = "uci-news-aggregator.csv"  # Replace with your CSV path
sentiments_path = "sentiments.json"

# Get sentiments
sentiments = get_sentiments(news_csv_path, sentiments_path)
print(f"Total sentiments collected: {len(sentiments)}")

