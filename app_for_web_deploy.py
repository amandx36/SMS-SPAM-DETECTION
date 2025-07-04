
import streamlit as st
import sys
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

ps = PorterStemmer()

def Transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)  # split the sentence into the list of words that has words punctuation

    # create the empty list and append the word and letter only not the special character we use inbuilt method isalnum()
    l = []
    for i in text:
        if (i.isalnum()):  # no need to write ==True if the statement is true than it do the forward work else did not do the forward work !!
            l.append(i)

    # now put the l into the text container !!
    text = l[:]

    # // now clear for not using the extra memory so that we can reuse it !! !!
    l.clear()

    # now i have to remove the special characters and also remove the stopwords !!
    for i in text:
        if i not in string.punctuation and i not in stopwords.words('english'):
            l.append(i)

    text = l[:]
    l.clear()

    # now remove the stem words !!
    # now convert into the stem words !!!
    for i in text:
        l.append(ps.stem(i))
    text = l[:]
    l.clear()

    # now give all the data in the list now need in the full words !! we use the join function
    return " ".join(text)

## loading the vectorizer  !!
vectorizer = pickle.load(open('vectorizer.pkl','rb'))

## loading the model that is random forest
model = pickle.load(open('model.pkl','rb'))

# now make a gui !!
st.set_page_config(page_title="SMS Spam Detection", page_icon="📩", layout="centered") 


st.title("SMS Spam Detection !!")

# added markdown for better instruction visibility
st.markdown("### Enter your SMS message below to check if it's **Spam or Not Spam**.")

# changed to text_area for bigger input box
text_input = st.text_area("✏Your Message:", height=150)

# three works
# 1 preprocess
# "2 vectorize "
# "3 predict "
# "4 display

if st.button('🔍 Predict'):
    # added check for empty input
    if text_input.strip() == "":
        st.warning("Enter a message to predict.")
    else:
        # 1 process the message !!
        transformed_input = Transform_text(text_input)

        # converted the input text into cleaned, stemmed text

        # vectorize
        sms_input_int_list = [transformed_input]
        vectorized_input = vectorizer.transform(sms_input_int_list)

        # vectorized input data for model

        # predict
        result = model.predict(vectorized_input)

        # predicted value is stored in result

        # display with icons and professional colors
        if result[0] == 1:
            st.error("Spam Detected!")   #is spam
        else:
            st.success(" NOT Spam Message")   # not spam
