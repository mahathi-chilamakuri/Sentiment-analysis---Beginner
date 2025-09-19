import pandas as pd
from io import StringIO
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download required NLTK data
nltk.download('vader_lexicon')

# Initialize VADER Sentiment Analyzer
sid = SentimentIntensityAnalyzer()

# --- TextBlob Dataset: 5 example tweets ---
tweets_csv = """
text
"I love the new design of your website!"
"The flight was delayed and it was so frustrating."
"Looking forward to the weekend :)"
"The customer service was okay, nothing special."
"I'm not sure how I feel about this new update."
"""

# Read tweets into DataFrame
tweets_df = pd.read_csv(StringIO(tweets_csv))

# TextBlob Sentiment Analysis on tweets
print("TextBlob Sentiment Analysis on Tweets:")
for i, tweet in enumerate(tweets_df['text']):
    analysis = TextBlob(tweet)
    polarity = analysis.sentiment.polarity
    subjectivity = analysis.sentiment.subjectivity
    print(f"\nTweet {i+1}: {tweet}")
    print(f"Polarity: {polarity:.3f}")
    print(f"Subjectivity: {subjectivity:.3f}")

# --- VADER Dataset: 5 example IMDB movie reviews ---
imdb_reviews = [
    "This movie was fantastic! I really enjoyed it.",
    "Worst film ever. I wasted two hours of my life.",
    "It was okay, not great but not terrible either.",
    "Absolutely loved the acting and the story.",
    "The plot was boring and predictable."
]

# VADER Sentiment Analysis on IMDB reviews
print("\n\nVADER Sentiment Analysis on IMDB Reviews:")
for i, review in enumerate(imdb_reviews):
    scores = sid.polarity_scores(review)
    compound = scores['compound']
    if compound >= 0.05:
        sentiment = 'Positive'
    elif compound <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    print(f"\nReview {i+1}: {review}")
    print(f"Compound Score: {compound:.3f}")
    print(f"Classified Sentiment: {sentiment}")
