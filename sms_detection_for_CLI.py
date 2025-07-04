

import sys

import pickle 



import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()




# make the function to convert into the lower case !! 

def Transform_text (text):
    text = text.lower() 
    text = nltk.word_tokenize(text)    # spilt the sentence into the list of words that has words punctutaion 
    #create the empty list and append the word and letter only not the special character we use inbuilt method isalnum()
    l = []
    for i in text :
        if (i.isalnum()):   # no need to right ==True if the statement id   true than it do the forward work  else did not do the forward work !! 
            
            l.append(i)

    # now put the l into the text container !! 
    text = l[:]

    # // now clear for not using the extra memeory  so that we can reuse it !! !!
    l.clear()

     # now i have to remove the special characters  and also remove the stopwords  !!

    for i in text :
        if i not in string.punctuation and i not in stopwords.words('english') :
            l.append(i)


    text = l[:]

    l.clear()

    # now remove the stem words !! 
  
    # now convert into the stem words !!! 
    for i in text :
        l.append(ps.stem(i))
    text = l[:]
    l.clear()

    # now give all the data in the list now need in the full words !! we use the join function 
    return " ".join(text)
  
            

## loading the vectorizer  !!


vectorizer = pickle.load(open('vectorizer.pkl','rb')) 


## loading the model that is random forest

model  = pickle.load(open('model.pkl','rb'))

# take the input from the user !! 

sms_input = input("Enter the message you want to check -->   ") 




# now apply the transformed text function into the input data !! 
transformed_input = Transform_text(sms_input)

# converting the input data into the list as the vectorizer need the data in the list 

sms_input_int_list = [transformed_input]

 
vectorized_input = vectorizer.transform(sms_input_int_list)


# predicted value is store in the  result  0  is for spam 
# and 1 for ham 


result  = model.predict(vectorized_input)

if result[0]==1:
   print("spam !")

else:
    print("The SMS is NOT SPAM  !!!  ")
 