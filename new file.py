import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

# Load data
data = pd.read_csv("spam.csv")  # Fixed: removed /content/ path

data.drop_duplicates(inplace=True)

data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])

mess = data['Message']
cat = data['Category']

mess_train, mess_test, cat_train, cat_test = train_test_split(mess, cat, test_size=0.2, random_state=42)

cv = CountVectorizer(stop_words='english')
features_train = cv.fit_transform(mess_train)
features_test = cv.transform(mess_test)

# Creating Model
model = MultinomialNB()
model.fit(features_train, cat_train)

# Predict function
def predict(message):
    input_message = cv.transform([message]).toarray()
    result = model.predict(input_message)
    return result

# Streamlit UI
st.header('Spam Detection')

input_messge = st.text_input('Enter Message Here')

if st.button('Validate'):
    output = predict(input_messge)   # Fixed: was input_mess (typo)
    st.text(str(output[0]))          # Fixed: was st.text = output
