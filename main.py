from urllib import response
import config
import tweepy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Authenticate with Twitter
client  = tweepy.Client(consumer_key=config.api_key,
                        consumer_secret=config.api_secret,
                        access_token=config.access_token,
                        access_token_secret=config.access_secret)

# Get the user's most recent tweet
tweet = 'Hello World! @realDonaldTrump'

#preprocess the tweet
tweet_words = []

for word in tweet.split(' '):
    if word.startswith('@') and len(word) > 1:
        word = '@user'
    elif word.startswith('http'):
        word = 'http'
    tweet_words.append(word)

tweet_processed = " ".join(tweet_words)

#load the NLP model and tokenizer (roberta)
roberta = "cardiffnlp/twitter-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)
lables = ['Negative', 'Neutral', 'Positive']

# Analyze the tweet
encoded_tweet = tokenizer(tweet_processed, return_tensors='pt')
output = model(**encoded_tweet)

scores = output[0][0].detach().numpy()
scores = softmax(scores)
for i in range(len(scores)):

    l = lables[i]
    s = scores[i]
    print(l, s)