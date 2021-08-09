# NLP Project

# Import relevant modules
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize

#nltk.download('punkt')

# Function to remove URLs from message
def remove_url(message):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', message)

# Function to remove HTML tags from message
def remove_html(message):
    html = re.compile(r'<.*?>')
    return html.sub(r'', message)

# Import SMS data into a pandas dataframe
df = pd.read_csv('clean_nus_sms.csv', index_col = 0)

# Used df.info() to find that there are 3 null entries in Message field

# Conducting text preprocessing
# Remove null entries
df = df.dropna()
# Normalise text by setting all messages to lowercase
df["clean_message"] = df["Message"].str.lower()
# Remove punctuation
df['clean_message'] = df['clean_message'].replace('[^\w\s]','')
# Remove URLs
df['clean_message'] = df['clean_message'].apply(lambda message: remove_url(message))
# Remove HTML tags
df['clean_message'] = df['clean_message'].apply(lambda message: remove_html(message))
# Tokenize messages
df['tokenized_message'] = df.apply(lambda x: nltk.word_tokenize(x['clean_message']), axis = 1)

print(df.head())
df.to_csv("preprocessed_clean_nus_sms.csv", header = True)