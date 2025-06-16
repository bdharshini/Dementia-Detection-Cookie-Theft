#install all required modules
pip install huggingface_hub
pip install datasets
pip install nltk
pip install SpeechRecognition
pip install PyAudio

from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from string import punctuation
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,classification_report
import seaborn as sns
import joblib
import speech_recognition as sr

#load dataset
ds = load_dataset("MearaHe/dementiabank")
print(ds)

#save dataset
df=ds['train'].to_pandas()
df.to_csv("data_dementia.csv")
print("Data saved to csv!!")
df.head()
df.shape

#plot class ditribution
plt.hist(df['output'],color='skyblue',edgecolor='black')
plt.xlabel('output')
plt.ylabel('number of people')
plt.title('frequency of output')
plt.savefig("class_distribution.png")
plt.show()

#Check the inst col
print(df['instruction'].unique())
#only one unique value present...drop
df = df.drop(columns=['instruction'])
df.head()

#convert output to numeric value with label encoding
label_encode=preprocessing.LabelEncoder()
df['output']=label_encode.fit_transform(df['output'])
df.head()
df['output'].unique()

#do not remove stop words...it may contain valuable data to detect
def preprocess_inp(sent):
    # sent is initially a string and nrmlly iterating on it would produce letters and not words thus split forst to a list
    sent=sent.split(" ") 
    clean=[word.lower() for word in sent if word not in punctuation]
    return ' '.join(clean)

df['input']=df['input'].apply(preprocess_inp)
df.head()


X_train,X_test,y_train,y_test=train_test_split(df['input'],df['output'],test_size=0.2,random_state=42)
print(X_train.shape,y_train.shape,y_test.shape,X_test.shape)

vect=TfidfVectorizer()
X_train_tfidf=vect.fit_transform(X_train)
X_test_tfidf=vect.transform(X_test)
print(X_train_tfidf,X_test_tfidf)

#models
from sklearn.linear_model import LogisticRegression
def logistic_reg():
    model=LogisticRegression()
    model.fit(X_train_tfidf,y_train)
    preds=model.predict(X_test_tfidf)
    return accuracy_score(preds,y_test)

from sklearn.tree import DecisionTreeClassifier
def classify_entropy():
    model=DecisionTreeClassifier(criterion="entropy",random_state=42)
    model.fit(X_train_tfidf,y_train)
    preds=model.predict(X_test_tfidf)
    return accuracy_score(preds,y_test)

from sklearn.tree import DecisionTreeClassifier
def classify_gini():
    model=DecisionTreeClassifier(criterion="gini",random_state=42)
    model.fit(X_train_tfidf,y_train)
    preds=model.predict(X_test_tfidf)
    return accuracy_score(preds,y_test)

from sklearn.svm import SVC
def classify_SVM():
    model=SVC()
    model.fit(X_train_tfidf,y_train)
    preds=model.predict(X_test_tfidf)
    train_pred=model.predict(X_train_tfidf)
    return accuracy_score(preds,y_test)

from xgboost import XGBClassifier
def classify_xgb():
    model=XGBClassifier()
    model.fit(X_train_tfidf,y_train)
    preds=model.predict(X_test_tfidf)
    return accuracy_score(preds,y_test)

#call all models
LR=logistic_reg()
DTCE=classify_entropy()
DTCG=classify_gini()
SVM=classify_SVM()
XGB=classify_xgb()

#plot all the models accuracy to find the best working model
report={
    "LogisticRegression":LR,
    "DecisionTreeGini":DTCG,
    "DecisionTreeEntrophy":DTCE,
    "SVC":SVM,
    "XGBoost":XGB,
}
print(report)
plt.figure(figsize=(12,6))
sns.barplot(x=report.keys(),y=report.values(),color='skyblue')
plt.title("Accuracy comparision betweeen models")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.savefig("model_accuracies.png")
plt.show()

#SVC was the best model
#save model and vect
model=SVC()
model.fit(X_train_tfidf,y_train)
joblib.dump(model,"model.pkl")
joblib.dump(vect,"vectorizer.pkl")


#get audio and predict
def speech_to_text():
    recognizer=sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak now..")
        audio=recognizer.listen(source)
        print("Audio recorded...Transcribing....")
    try:
        text=recognizer.recognize_google(audio)
        print(f"Transcribed text:{text}")
        return text
    except sr.UnknownValueError:
        print("Could not undertstand audio...")
        return ""
    except sr.RequestError:
        print("An error occured...")
        return ""

text=speech_to_text()
if text:
            text=preprocess_inp(text)
            text_vect=vect.transform([text])
            model=joblib.load("model.pkl")
            prediction=model.predict(text_vect)
            if prediction ==1:
                print("Oh no....U might want to schedule a check up\nPotential Dementia detected!!")
            else:
                print("No Dementia detected!!")

