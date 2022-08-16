from urllib import response
import config
import pandas as pd
import tweepy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Authenticate with Twitter
auth = tweepy.OAuthHandler(config.api_key, config.api_secret)
auth.set_access_token(config.access_token, config.access_secret)

# Create API object
api = tweepy.API(auth)

# Search for Tweets

keywords = 'stock market'
limit = 100

tweets = tweepy.Cursor(api.search_tweets, q=keywords, count=100, tweet_mode='extended').items(limit)


# Create a dataframe with the Tweets
columns = ['User', 'Tweet']
data = []

for tweet in tweets:
    data.append([tweet.user.screen_name, tweet.full_text])

df = pd.DataFrame(data, columns=columns)


#load the NLP model and tokenizer (roberta)
roberta = "cardiffnlp/twitter-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)
lables = ['Negative', 'Neutral', 'Positive']
scores_totals = [0, 0, 0]


# Running through dataframe and predicting sentiment
for index, row in df.iterrows():

    #preprocess the tweet
    tweet_words = []
    tweet = row['Tweet']
    for word in tweet.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = 'http'
        tweet_words.append(word)

    tweet_processed = " ".join(tweet_words)

    # Analyze the tweet
    encoded_tweet = tokenizer(tweet_processed, return_tensors='pt')
    output = model(**encoded_tweet)

    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    for i in range(len(scores)):

        scores_totals[i] += scores[i]

for i in range(len(scores_totals)):
    scores_totals[i] = scores_totals[i] / limit
    print(lables[i] + ": " + str(scores_totals[i]))

for i in range(len(scores_totals)):
    if scores_totals[i] > 0.5:
        print("The past 100 tweets relating to the Stock Market are mostly " + lables[i])
        