from flask import Flask, render_template
import pickle
import sklearn
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split
import pandas as pd

#Load data
def load_data():
    data = pd.read_excel('static/TweetDataUpdate.xlsx')
    return data

tweet_df = load_data()
df  = pd.DataFrame(tweet_df[['text','class']])

tfidf_vectorizer=TfidfVectorizer(use_idf=True) 
tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(tweet_data)

# get the first vector out (for the first document) 
first_vector_tfidfvectorizer=tfidf_vectorizer_vectors[0]

# place tf-idf values in a pandas data frame
hasil = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"]) 
hasil.sort_values(by=["tfidf"],ascending=False)

#Split Train & Test
X_train, X_test, y_train ,y_test = train_test_split(tfidf_vectorizer_vectors,sentiment,train_size = .8 , test_size = .2 , random_state = 0)

filename = 'static/finalized_model.sav'

#Import trained model
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/about')
def about():
    return render_template('about.html') 

@app.route('/preprocessing')
def preprocessing():
    return render_template('preprocessing.html')

@app.route('/vectorizer')
def vectorizer():
    return render_template('vectorizer.html') 

@app.route('/training')
def training():
    return render_template('training.html') 

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')  

if __name__ == '__main__':
    app.run(debug=True)