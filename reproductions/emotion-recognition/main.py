import tweepy
import json
import emoji

class ApiConfiguration:
    def __init__(self, configuration_json):
        self.consumer_key = configuration_json['consumer_key']
        self.consumer_secret = configuration_json['consumer_secret']
        self.access_token = configuration_json['access_token']
        self.access_token_secret = configuration_json['access_token_secret']

with open('./configuration.json', 'r') as file:
    configuration = json.load(file)

apiConfig = ApiConfiguration(configuration)

auth = tweepy.OAuthHandler(consumer_key=apiConfig.consumer_key, consumer_secret=apiConfig.consumer_secret)
auth.set_access_token(apiConfig.access_token, apiConfig.access_token_secret)

api = tweepy.API(auth)

public_tweets = api.home_timeline()
print("Read home timeline as a access test - tweets amount {}".format(len(public_tweets)))
for tweet in public_tweets:
    print(tweet.text)

KEYWORDS = ['joy', 'angry', 'sad', 'fun', 'love', 'happiness', 'happy', 'excited']
OUTPUT_FILE = './tweets.txt'
TWEETS_TO_CAPTURE = 300

class TweeterStreamListener(tweepy.StreamListener):
    def __init__(self, api=None):
        super(TweeterStreamListener, self).__init__()
        self.num_tweets = 0
        self.file = open(OUTPUT_FILE, 'w')

    def on_status(self, status):
        tweet = status._json

        self.file.write(json.dumps(tweet) + '\n')
        self.num_tweets += 1

        if self.num_tweets <= TWEETS_TO_CAPTURE:
            if self.num_tweets % 100 == 0:
                print("Number of tweets caputerd with given keywords {}".format(self.num_tweets))
            return  True
        else:
            return False
        self.file.close()

    def on_error(self, status_code):
        print(status_code)

#ts = TweeterStreamListener(api)

#stream = tweepy.Stream(auth, ts)
#stream.filter(track=KEYWORDS)

with open(OUTPUT_FILE) as tweets_file:
    for line in tweets_file:
        tweet = json.loads(line)
        print(emoji.emojize(tweet['text']))
