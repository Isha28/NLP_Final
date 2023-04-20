import tweepy

# Add your Twitter API credentials
consumer_key = "..."
consumer_secret = "..."
access_token = "..."
access_token_secret = "..."

# Authenticate to Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Create an API object
api = tweepy.API(auth)

# Define the search query
search_words = "protestors"
date_since = "2023-04-15"

# Collect tweets using the search query
tweets = tweepy.Cursor(api.search_tweets, q=search_words, lang="en", since_id=date_since).items()

# Print the text of the retrieved tweets
for tweet in tweets:
    print(tweet.text)