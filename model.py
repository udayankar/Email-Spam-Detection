import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sea
import nltk
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import  GaussianNB , MultinomialNB , BernoulliNB
from sklearn.metrics import accuracy_score , confusion_matrix , precision_score  
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
import string
string.punctuation
nltk.download("punkt")
from wordcloud import WordCloud
from collections import Counter
import joblib



#Accessing dataset
data_set = pd.read_csv("email.csv")
data_set_df = pd.DataFrame(data_set)
# print(data_set)
# print(data_set_df)

#Finding null values
print(data_set_df.isnull().sum())
#Finding duplicate values
print(data_set_df.duplicated().sum())

#Removing duplicate values
data_set_df = data_set_df.drop_duplicates(keep = "first")

#Finding unwanted value
# print(data_set_df["Category"])

# #Removing unwanted value
data_set_df = data_set_df.iloc[:-1]

# #EDA
# plt.pie(data_set_df["Category"].value_counts() , labels = data_set_df["Category"].value_counts().index , autopct = "%0.2f")
# plt.show()

data_set_df["num_characters"] = data_set_df["Message"].apply(len)

data_set_df["num_words"] = data_set_df["Message"].apply(lambda x : len( nltk.word_tokenize(x)))

data_set_df["num_sentences"] = data_set_df["Message"].apply(lambda x : len( nltk.sent_tokenize(x)))

# print(data_set_df[["num_characters" , "num_words" , "num_sentences"]].describe())

# print(data_set_df[data_set_df["Category"] == "ham"][["num_characters" , "num_words" , "num_sentences"]].describe())

# print(data_set_df[data_set_df["Category"] == "spam"][["num_characters" , "num_words" , "num_sentences"]].describe())

# sea.histplot(data_set_df[data_set_df["Category"] == "ham"]["num_characters"] , color = "green")
# sea.histplot(data_set_df[data_set_df["Category"] == "spam"]["num_characters"] , color = "red")
# plt.legend(loc = "upper right" , labels = ["Green = Ham" , "Red = Spam"])
# plt.show()

# sea.histplot(data_set_df[data_set_df["Category"] == "ham"]["num_words"] , color = "green")
# sea.histplot(data_set_df[data_set_df["Category"] == "spam"]["num_words"] , color = "red")
# plt.legend(loc = "upper right" , labels = ["Green = Ham" , "Red = Spam"])
# plt.show()

# sea.histplot(data_set_df[data_set_df["Category"] == "ham"]["num_sentences"] , color = "green")
# sea.histplot(data_set_df[data_set_df["Category"] == "spam"]["num_sentences"] , color = "red")
# plt.legend(loc = "upper right" , labels = ["Green = Ham" , "Red = Spam"])
# plt.show()

#Data Preproccesing
def transform_text (text) :
    y = []
    text = text.lower()
    text = nltk.word_tokenize(text)
    for i in text :
        if i.isalnum() and i not in ENGLISH_STOP_WORDS and i not in string.punctuation:
            y.append(ps.stem(i))
    
    return " ".join(y)
data_set_df["Transformed-Text"] = data_set_df["Message"].apply(transform_text)

wc = WordCloud( width = 500 , height = 500 , min_font_size = 10 , background_color = "white" )

ham_wc = wc.generate(data_set_df[data_set_df["Category"] == "ham"]["Transformed-Text"].str.cat(sep=""))
plt.imshow(ham_wc)
plt.axis("off")
plt.title("Ham word cloud")
plt.show()

spam_wc = wc.generate(data_set_df[data_set_df["Category"] == "spam"]["Transformed-Text"].str.cat(sep=""))
plt.imshow(spam_wc)
plt.axis("off")
plt.title("Spam word cloud")
plt.show()

spam_corpus = []
for msg in data_set_df[data_set_df["Category"] == "spam"]["Transformed-Text"].to_list() :
    for word in msg.split() :
        spam_corpus.append(word)
spam_word_df = pd.DataFrame(Counter(spam_corpus).most_common(50))
sea.barplot(x = spam_word_df[0] ,y = spam_word_df[1])
plt.xticks(rotation="vertical")
plt.title("Most used words in Spam")
plt.show()

ham_corpus = []
for msg in data_set_df[data_set_df["Category"] == "ham"]["Transformed-Text"].to_list() :
    for word in msg.split() :
        ham_corpus.append(word)
ham_word_df = pd.DataFrame(Counter(ham_corpus).most_common(50))
sea.barplot(x = ham_word_df[0] ,y = ham_word_df[1])
plt.xticks(rotation="vertical")
plt.title("Most used words in Ham")
plt.show()

# #Model Building
data_set_df["Category"] = data_set_df["Category"].replace({"ham" : 0 , "spam" : 1})
tf = TfidfVectorizer()
X = tf.fit_transform(data_set_df["Transformed-Text"]).toarray()
Y = data_set_df["Category"].values
X_train , X_test , Y_train , Y_test  = train_test_split(X , Y , test_size = 0.8 , random_state = 2)

gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

gnb.fit(X_train , Y_train)
Y_pred1 = gnb.predict(X_test)
print(accuracy_score(Y_test , Y_pred1))
print(confusion_matrix(Y_test , Y_pred1))
print(precision_score(Y_test , Y_pred1))

mnb.fit(X_train , Y_train)
Y_pred2 = mnb.predict(X_test)
print(accuracy_score(Y_test , Y_pred2))
print(confusion_matrix(Y_test , Y_pred2))
print(precision_score(Y_test , Y_pred2))

bnb.fit(X_train , Y_train)
Y_pred3 = bnb.predict(X_test)
print(accuracy_score(Y_test , Y_pred3))
print(confusion_matrix(Y_test , Y_pred3))
print(precision_score(Y_test , Y_pred3))

#Saving Model
joblib.dump(mnb, "spam_model.pkl")
joblib.dump(tf, "vectorizer.pkl")