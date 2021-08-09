# SMS Analysis

#Import dataframe from preprocessing
from nltk.corpus.reader.chasen import test
from sms_preprocess import df, remove_html, remove_url

#Import relevant modules
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import twitter_samples
import random

# Download twitter data and models for sentiment analysis
#nltk.download('twitter_samples')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')

# Function that identified when a feature is present in message
def document_feature(document, sentiment_features):
    document_tokens = set(document)
    features = {}
    for token in sentiment_features:
        features[token] = (token in document_tokens)
    return features

# Function that converts sentiment into a binary value
def convert_sentiment(sentiment):
    if sentiment == 'Positive':
        return 1
    else:
        return -1

# Used .value_counts() method to find that countries were repeated
# Replace country codes with countries to reduce duplications
df = df.replace({'country': {
    'SG': 'Singapore',
    'USA': 'United States',
    'india': 'India',
    'INDIA': 'India',
    'srilanka': 'Sri Lanka',
    'UK': 'United Kingdom',
    'BARBADOS': 'Barbados',
    'Italia': 'Italy',
    'jamaica': 'Jamaica',
    'MY': 'Malaysia',
    'unknown': 'Unknown'
}})

'''
country_value_counts = df['country'].value_counts()
top_10_country_counts = country_value_counts.head(10)
top_10_country_counts.plot.barh()
plt.savefig("top_10_countries_value_counts.png")
plt.show()
'''

# Instantiate twitter samples
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

# Create tokens from tweets
pos_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
neg_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

# Clean tokens
clean_pos_tweets = []
clean_neg_tweets = []

for tweet in pos_tweet_tokens:
    clean_tweet = []
    for token in tweet:
        cleaned_token = remove_url(token)
        cleaned_token = remove_html(cleaned_token)
        clean_tweet.append(cleaned_token)
    clean_pos_tweets.append(clean_tweet)

for tweet in neg_tweet_tokens:
    clean_tweet = []
    for token in tweet:
        cleaned_token = remove_url(token)
        cleaned_token = remove_html(cleaned_token)
        clean_tweet.append(cleaned_token)
    clean_neg_tweets.append(clean_tweet)

all_tweets = clean_pos_tweets + clean_neg_tweets
all_tweet_tokens = [token for sublist in all_tweets for token in sublist]

# Make frequency distribution
all_tokens = nltk.FreqDist(token for token in all_tweet_tokens)

sentiment_features = [word for (word, freq) in all_tokens.most_common(10000)]

pos_labeled_tweets = [(tweet, "Positive") for tweet in clean_pos_tweets]
neg_labeled_tweets = [(tweet, "Negative") for tweet in clean_neg_tweets]
all_labeled_tweets = pos_labeled_tweets + neg_labeled_tweets

random.seed(42)
# Shuffle tweets
random.shuffle(all_labeled_tweets)
# Create a list of (token, sentiment) pairs for all features in tweet
feature_set = [(document_feature(doc, sentiment_features), s) for (doc, s) in all_labeled_tweets]
training_set, testing_set = feature_set[:7000], feature_set[7000:]

# Accuracy for tweets was 99.6%
classifier = nltk.NaiveBayesClassifier.train(training_set)
accuracy = nltk.classify.accuracy(classifier, testing_set)
#print('Accuracy: ', accuracy)
#print(classifier.show_most_informative_features(20))

sentiments = []
for message in df['tokenized_message']:
    sentiments.append(str((classifier.classify(dict([token, True] for token in message)))))
df['Sentiments'] = sentiments

#sns.countplot(x = 'Sentiments', data = df, palette = 'RdBu')
#plt.savefig('sentiment_analysis.png')
#plt.show()

df['Sentiment_Score'] = df['Sentiments'].apply(convert_sentiment)

# Create new dataframe for sentiments grouped by country and aggregated by mean
sentiment_df = df.groupby(['country']).mean()
sentiment_df.reset_index(inplace = True)

fig, ax = plt.subplots(figsize = (40,15))
sns.barplot(x = 'country', y = 'Sentiment_Score', data = sentiment_df, ax = ax, color = 'lightseagreen')
plt.savefig('sentiment_by_country.png')
plt.show()