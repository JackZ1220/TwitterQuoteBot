from urllib import response
import config
import tweepy
import schedule
import time
import requests
from quoters import Quote


client  = tweepy.Client(consumer_key=config.api_key,
                        consumer_secret=config.api_secret,
                        access_token=config.access_token,
                        access_token_secret=config.access_secret)

def tweet_job():
    quote = Quote.print()
    response = client.create_tweet(text=quote)

tweet_job()
print(response)
#schedule.every().day.at("08:00").do(tweet_job)