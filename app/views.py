import re
import json
import nltk
import gensim
import numpy as np
import pandas as pd
from io import StringIO
from sklearn.svm import SVC
# import PIL
import seaborn as sns
import pytesseract
import matplotlib.pyplot as plt
from django.shortcuts import render
from gensim.models import KeyedVectors
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.ensemble import VotingClassifier
from multiprocessing.sharedctypes import Value
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# from gensim.models.fasttext import load_facebook_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
global score


def index(request):
    context = {"a": "Hello World"}
    return render(request, "index.html", context)


def home(request):
    context = {"a": "Hello World"}
    return render(request, "home.html", context)


def tokenization(text):
    tokens = re.split("W+", text)
    return tokens


def remove_stopwords(text):
    tokens = [token for token in text if not token in nltk_stopwords]
    return tokens



def stemming(text):
    stem_text = [porter_stemmer.stem(word) for word in text]
    stem_text = " ".join(stem_text)
    return stem_text


def lemmatizer(text):
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    lemm_text = " ".join(lemm_text)
    return lemm_text

#Removes Punctuations-----------1
def remove_punct(data):
    punct_tag=re.compile(r'[^\w\s]')
    data=punct_tag.sub(r'',data)
    return data


# Remove whitespace ----2
def remove_whitespace(data):
    tag=re.compile(r'\s+')
    data=tag.sub(r' ',data)
    return data

#Removes Roman words----------3
def remove_roman(data):
    en_tag =re.compile(r'^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$')
    data=en_tag.sub(r'',data)
    return data


#Remove redundant words---------------4
def remove_redun(data):
    red_tag=re.compile(r'[?<=(  )\\]|[&&|\|\|-]')
    data=red_tag.sub(r' ',data)
    return "".join(data)


#Removes Numbers -------5
def remove_num(data):
    tag=re.compile(r'[0-9]+')
    data=tag.sub(r'',data)
    return data


def createfile():
    global f
    l = []
    l.extend(X_train_vtc)
    l.extend(X_test_vtc)
    f = pd.DataFrame(l)
    f["labels"] = Y
    f.to_csv("./media/AI_ready_data.csv", index=False)
    f = "AI_ready_data.csv"


def tf_idf():
    global nltk_stopwords, porter_stemmer, wordnet_lemmatizer, X, Y, score, df, X_train, X_test, y_train, y_test, X_train_vtc, X_test_vtc
    # X = X.apply(lambda x: tokenization(x))
    # X = X.apply(lambda x: remove_stopwords(x))
    # X = X.apply(lambda x: stemming(x))
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    X_train_vtc = X_train
    X_test_vtc = X_test
    createfile()


def count_vect():
    global nltk_stopwords, porter_stemmer, wordnet_lemmatizer, X, Y, score, df, X_train, X_test, y_train, y_test, X_train_vtc, X_test_vtc
    # X = X.apply(lambda x: tokenization(x))
    # X = X.apply(lambda x: remove_stopwords(x))
    # X = X.apply(lambda x: stemming(x))
    cv = CountVectorizer()
    x = cv.fit_transform(X, Y)
    X_train, X_test, y_train, y_test = train_test_split(x, Y, test_size=0.3)
    X_train_vtc = X_train
    X_test_vtc = X_test
    createfile()


def word2vec_customized():
    global nltk_stopwords, porter_stemmer, wordnet_lemmatizer, X, Y, score, df, X_train, X_test, y_train, y_test, X_train_vtc, X_test_vtc
    X_train_vect_avg = []
    X_test_vect_avg = []
    # X = X.apply(lambda x: tokenization(x))
    # X = X.apply(lambda x: remove_stopwords(x))
    # X = X.apply(lambda x: stemming(x))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    w2v_model = gensim.models.Word2Vec(X_train, vector_size=100, window=5, min_count=2)
    words = set(w2v_model.wv.index_to_key)
    X_train_vect = np.array(
        [np.array([w2v_model.wv[i] for i in ls if i in words]) for ls in X_train]
    )
    X_test_vect = np.array(
        [np.array([w2v_model.wv[i] for i in ls if i in words]) for ls in X_test]
    )
    X_train_vect_avg = []
    for v in X_train_vect:
        if v.size:
            X_train_vect_avg.append(v.mean(axis=0))
        else:
            X_train_vect_avg.append(np.zeros(100, dtype=float))
    X_test_vect_avg = []
    for v in X_test_vect:
        if v.size:
            X_test_vect_avg.append(v.mean(axis=0))
        else:
            X_test_vect_avg.append(np.zeros(100, dtype=float))
    X_train_vtc = X_train_vect_avg
    X_test_vtc = X_test_vect_avg
    createfile()
    y_train = y_train.values.ravel()
    X_train = X_train_vect_avg
    X_test = X_test_vect_avg


def Word2vec_google():
    global nltk_stopwords, porter_stemmer, wordnet_lemmatizer, X, Y, score, df, X_train, X_test, y_train, y_test, X_train_vtc, X_test_vtc
    X_train_vect_avg = []
    X_test_vect_avg = []
    # X = X.apply(lambda x: tokenization(x))
    # X = X.apply(lambda x: remove_stopwords(x))
    # X = X.apply(lambda x: stemming(x))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    filename = "./media/GoogleNews-vectors-negative300.bin"
    model_g = KeyedVectors.load_word2vec_format(filename, binary=True)
    words = set(model_g.index_to_key)
    X_train_vect = np.array(
        [np.array([model_g[i] for i in ls if i in words]) for ls in X_train]
    )
    X_test_vect = np.array(
        [np.array([model_g[i] for i in ls if i in words]) for ls in X_test]
    )
    for v in X_train_vect:
        if v.size:
            X_train_vect_avg.append(v.mean(axis=0))
        else:
            X_train_vect_avg.append(np.zeros(300, dtype=float))
    for v in X_test_vect:
        if v.size:
            X_test_vect_avg.append(v.mean(axis=0))
        else:
            X_test_vect_avg.append(np.zeros(300, dtype=float))
    X_train_vtc = X_train_vect_avg
    X_test_vtc = X_test_vect_avg
    createfile()
    y_train = y_train.values.ravel()
    X_train = X_train_vect_avg
    X_test = X_test_vect_avg


# def fasttext():
#     global nltk_stopwords,porter_stemmer,wordnet_lemmatizer,X,Y,score,df,X_train, X_test, y_train, y_test,X_train_vtc,X_test_vtc
#     X=X.apply(lambda x:tokenization(x))
#     X=X.apply(lambda x:remove_stopwords(x))
#     X=X.apply(lambda x:stemming(x))
#     X_train_vect_avg= []
#     X_test_vect_avg = []
#     X_train, X_test, y_train, y_test = train_test_split (X,Y)
#     filename =r"C:/Users/nikhil/Desktop/cc.en.300.bin"
#     model_g= load_facebook_model(filename)
#     words =set(model_g.wv.index_to_key)
#     X_train_vect= np.array([np.array([model_g.wv[i] for i in ls if i in words])
#                          for ls in X_train])
#     X_test_vect = np.array([np.array([model_g.wv[i] for i in ls if i in words])
#                          for ls in X_test])
#     for v in X_train_vect:
#         if v.size:
#             X_train_vect_avg.append(v.mean(axis=0))
#         else:
#             X_train_vect_avg.append(np.zeros(300, dtype=float))
#     for v in X_test_vect:
#         if v.size:
#             X_test_vect_avg.append(v.mean(axis=0))
#         else:
#             X_test_vect_avg.append(np.zeros(300, dtype=float))
#     X_train_vtc=X_train_vect_avg
#     X_test_vtc=X_test_vect_avg
#     createfile()
#     y_train=y_train.values.ravel()
#     X_train=X_train_vect_avg
#     X_test=X_test_vect_avg


def glove_w():
    global glove
    global nltk_stopwords, porter_stemmer, wordnet_lemmatizer, X, Y, score, df, X_train, X_test, y_train, y_test, X_train_vtc, X_test_vtc
    glove = {}
    # X = X.apply(lambda x: tokenization(x))
    # X = X.apply(lambda x: remove_stopwords(x))
    # X = X.apply(lambda x: stemming(x))
    total_vocabulary = set(word for text in X for word in text)
    with open(r"./media/glove.twitter.27B.100.txt", "rb") as f:
        for line in f:
            parts = line.split()
            word = parts[0].decode("utf-8")
            if word in total_vocabulary:
                vector = np.array(parts[1:], dtype=np.float32)
                glove[word] = vector

    class W2vVectorizer(object):
        def __init__(self, w2v):
            self.w2v = w2v
            if len(w2v) == 0:
                self.dimensions = 0
            else:
                self.dimensions = len(w2v[next(iter(glove))])

        def fit(self, X, Y):
            return self

        def transform(self, X):
            return np.array(
                [
                    np.mean(
                        [self.w2v[w] for w in words if w in self.w2v]
                        or [np.zeros(self.dimensions)],
                        axis=0,
                    )
                    for words in X
                ]
            )

    vectorizer = W2vVectorizer(glove)
    X_glove = vectorizer.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_glove, Y, test_size=0.2)
    X_train_vtc = X_train
    X_test_vtc = X_test
    createfile()


def RandomForest_classifier():
    rf_rf = RandomForestClassifier()
    rf_model_rf = rf_rf.fit(X_train, y_train)
    y_pred = rf_model_rf.predict(X_test)
    score = classification_report(y_test, y_pred, output_dict=True)
    return score


def decision_tree_classifier():
    rf_dt = DecisionTreeClassifier()
    rf_model_dt = rf_dt.fit(X_train, y_train)
    y_pred = rf_model_dt.predict(X_test)
    score = classification_report(y_test, y_pred, output_dict=True)
    return score


def KNeighbors_classifier():
    rf_kn = KNeighborsClassifier()
    rf_model_kn = rf_kn.fit(X_train, y_train)
    y_pred = rf_model_kn.predict(X_test)
    score = classification_report(y_test, y_pred, output_dict=True)
    return score


def Logistic_Regression_classifier():
    clf = LogisticRegression().fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = classification_report(y_test, y_pred, output_dict=True)
    return score


def SVC_classsifier():
    rf_svc = SVC()
    rf_model_svc = rf_svc.fit(X_train, y_train)
    y_pred = rf_model_svc.predict(X_test)
    score = classification_report(y_test, y_pred, output_dict=True)
    return score


def stacking():
    estimator_list = [
        ("knn", KNeighborsClassifier()),
        ("svm_rbf", SVC()),
        ("dt", DecisionTreeClassifier()),
        ("rf", RandomForestClassifier()),
        ("mlp", MLPClassifier(alpha=1, max_iter=1000)),
    ]
    stack_model = StackingClassifier(
        estimators=estimator_list, final_estimator=LogisticRegression()
    )
    stack_model.fit(X_train, y_train)
    y_pred = stack_model.predict(X_test)
    score = classification_report(y_test, y_pred, output_dict=True)
    return score


def voting():
    global vcl
    if vcl == "V_RANDOM_CL":
        score = v_RandomForest_classifier()
    elif vcl == "V_DECISION_CL":
        score = v_decision_tree_classifier()
    elif vcl == "V_KNN_CL":
        score = v_KNeighbors_classifier()
    elif vcl == "V_SVC_CL":
        score = v_SVC_classsifier()
    elif vcl == "V_all":
        score = v_all()
    return score


def v_RandomForest_classifier():
    models = list()
    models.append(("dt1", RandomForestClassifier(max_depth=2)))
    models.append(("dt3", RandomForestClassifier(max_depth=3)))
    models.append(("dt5", RandomForestClassifier(max_depth=4)))
    models.append(("dt7", RandomForestClassifier(max_depth=5)))
    models.append(("dt9", RandomForestClassifier(max_depth=6)))
    ensemble = VotingClassifier(estimators=models, voting="soft")
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    score = classification_report(y_test, y_pred, output_dict=True)
    return score


def v_decision_tree_classifier():
    models = list()
    models.append(("dt1", DecisionTreeClassifier(max_depth=2)))
    models.append(("dt3", DecisionTreeClassifier(max_depth=3)))
    models.append(("dt5", DecisionTreeClassifier(max_depth=4)))
    models.append(("dt7", DecisionTreeClassifier(max_depth=5)))
    models.append(("dt9", DecisionTreeClassifier(max_depth=6)))
    ensemble = VotingClassifier(estimators=models, voting="soft")
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    score = classification_report(y_test, y_pred, output_dict=True)
    return score


def v_KNeighbors_classifier():
    models = list()
    models.append(("knn1", KNeighborsClassifier(n_neighbors=1)))
    models.append(("knn3", KNeighborsClassifier(n_neighbors=3)))
    models.append(("knn5", KNeighborsClassifier(n_neighbors=5)))
    models.append(("knn7", KNeighborsClassifier(n_neighbors=7)))
    models.append(("knn9", KNeighborsClassifier(n_neighbors=9)))
    ensemble = VotingClassifier(estimators=models, voting="soft")
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    score = classification_report(y_test, y_pred, output_dict=True)
    return score


def v_SVC_classsifier():
    models = list()
    models.append(("svm1", SVC(probability=True, kernel="poly", degree=1)))
    models.append(("svm2", SVC(probability=True, kernel="poly", degree=2)))
    models.append(("svm3", SVC(probability=True, kernel="poly", degree=3)))
    models.append(("svm4", SVC(probability=True, kernel="poly", degree=4)))
    models.append(("svm5", SVC(probability=True, kernel="poly", degree=5)))
    ensemble = VotingClassifier(estimators=models, voting="soft")
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    score = classification_report(y_test, y_pred, output_dict=True)
    return score


def v_all():
    models = [
        ("knn", KNeighborsClassifier()),
        ("svm_rbf", SVC(probability=True)),
        ("dt", DecisionTreeClassifier()),
        ("rf", RandomForestClassifier()),
        ("mlp", MLPClassifier(alpha=1, max_iter=1000)),
    ]
    ensemble = VotingClassifier(estimators=models, voting="soft")
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    score = classification_report(y_test, y_pred, output_dict=True)
    return score

# def simple_nn():
#     model = Sequential()
#     model.add(Dense(100, input_dim=100, activation='relu'))
#     model.add(Dense(52, activation='softmax'))
#     model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     model.fit(X_train, y_train, epochs=5, batch_size=100, validation_data=(X_test, y_test))
#     y_pred = model.predict(X_test)
#     score = classification_report(y_test, y_pred, output_dict=True)
#     return score


def if_text(uploaded_file,sep):
    global df
    uploaded_file=uploaded_file.read()
    uploaded_file = str(uploaded_file, "utf-8")
    uploaded_file = StringIO(uploaded_file)
    df = pd.read_csv(uploaded_file, sep, header=None, names=["label", "text"], on_bad_lines='skip')
    return df

def if_pdf():
    global df
    import PyPDF2
    l=[]    
    pdfFileObj=uploaded_file
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
    n=pdfReader.getNumPages()
    for i in range(n):
        pageObj = pdfReader.getPage(i)
        p = pageObj.extractText().split('\n')
        l.extend(p)
    labels=['student','project','course','faculty']
    n=len(l)
    s=' '
    df=pd.DataFrame({})
    for i in range(n):
        k=l[i].split()
        try:
            k=k[0]
        except:
            continue
        if k not in labels or i==0:
            s+=l[i]
        else:
            s=s.split()
            try:
                la=s[0]
            except:
                continue
            d=' '.join(s[1:])
            dict1 = {'label':[la],
            'text':d}
            dict1=pd.DataFrame(dict1)
            df=pd.concat([df,dict1], ignore_index = True)
            s=' '
        if k in labels:
            s+=l[i]
    s=s.split()
    la=s[0]
    d=' '.join(s[1:])
    dict1 = {'label':[la],
    'text':d}
    dict1=pd.DataFrame(dict1)
    df=pd.concat([df,dict1], ignore_index = True)
    s=' '
    return df

def if_image():
    global df
    import os
    img= PIL.Image.open(uploaded_file)
    pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files (x86)/Tesseract-OCR/tesseract'
    TESSDATA_PREFIX = "C:\Program Files (x86)\Tesseract-OCR"
    text_data = pytesseract.image_to_string(img.convert('RGB'), lang='eng')
    f= open("sample.txt", "w")
    f.write(text_data)
    f.close()
    with open('./sample.txt') as f:
        lines = f.readlines()
    labels=['student','project','course','faculty','Student','Project','Course','Faculty']
    n=len(lines)
    s=' '
    df=pd.DataFrame({})
    for i in range(n):
        k=lines[i].split()
        try:
            k=k[0]
        except:
            continue
        if k not in labels or i==0:
            s+=lines[i]
        else:
            s=s.split()
            try:
                la=s[0]
            except:
                continue
            d=' '.join(s[1:])
            dict1 = {'label':[la],
            'text':d}
            dict1=pd.DataFrame(dict1)
            df=pd.concat([df,dict1], ignore_index = True)
            s=' '
        if k in labels:
            s+=lines[i]
    s=s.split()
    la=s[0]
    d=' '.join(s[1:])
    dict1 = {'label':[la],
    'text':d}
    dict1=pd.DataFrame(dict1)
    df=pd.concat([df,dict1], ignore_index = True)
    return df




def preprocess(request):
    global nltk_stopwords, porter_stemmer, wordnet_lemmatizer, X, Y, score, df, vcl,uploaded_file
    score = ""
    nltk_stopwords = nltk.corpus.stopwords.words("english")
    f=""
    porter_stemmer = PorterStemmer()
    wordnet_lemmatizer = WordNetLemmatizer()
    uploaded_file=" "
    try:
        uploaded_file = request.FILES["uploaded_file"]
        print(uploaded_file)
    except:    
        if uploaded_file==" ":
            context={'f':"PLEASE UPLOAD A FILE!!!!"}
            return render(request, "table.html", context)

    # uploaded_file = str(uploaded_file, "utf-8")
    # uploaded_file = StringIO(uploaded_file)

    sep = str(request.POST["sep"])
    if uploaded_file.content_type=='image/jpeg':
        df=if_image()
    elif uploaded_file.content_type=='application/pdf':
        df=if_pdf()   
    elif uploaded_file.content_type=='text/plain':
        df=if_text(uploaded_file,sep)
    we,cl,pr='','',''
    we="TFIDF"
    cl="LOGISTIC_REGRESSION_CL"
    we = str(request.POST["word_embedding"])
    cl = str(request.POST["classifier"])
    if cl == "VOTING_CL":
        vcl="V_all"
        vcl = str(request.POST["votingin"])
    sep = "	"
    # df = pd.read_csv(uploaded_file, sep, header=None, names=["label", "text"])
    df = df.fillna("")
    X = df["text"]
    Y = df["label"]
    pr=["6","7","8"]
    pr=(request.POST["preprocessingg"])
    if "7" in pr:
        X = X.apply(lambda x: remove_stopwords(x)) 
    elif "1" in pr:
        X = X.apply(lambda x: remove_punct(x))
    elif "3" in pr:
        X = X.apply(lambda x: remove_roman(x))
    elif "4" in pr:
        X = X.apply(lambda x: remove_redun(x))
    elif "5" in pr:
        X = X.apply(lambda x: remove_punct(x))
    elif "2" in pr:
        X = X.apply(lambda x: remove_whitespace(x))   
    elif "8" in pr:  
        X = X.apply(lambda x: stemming(x))

    if we == "TFIDF":
        tf_idf()
    elif we == "COUNTVECT":
        count_vect()
    elif we == "WORDTOVECCOS":
        word2vec_customized()
    elif we == "WORDTOVECGOG":
        Word2vec_google()
    # elif we == "FAST-TEXT":
    #     print("fastext")
        # fasttext()
    elif we == "GLOVE":
        glove_w()

    if cl == "RANDOM_CL":
        score = RandomForest_classifier()
    elif cl == "DECISION_CL":
        score = decision_tree_classifier()
    elif cl == "KNN_CL":
        score = KNeighbors_classifier()
    elif cl == "LOGISTIC_REGRESSION_CL":
        score = Logistic_Regression_classifier()
    elif cl == "SVC_CL":
        score = SVC_classsifier()
    elif cl == "STACKING_CL":
        score = stacking()
    elif cl == "VOTING_CL":
        score = voting()
    elif cl == "simple_nn":
        score = simple_nn()
    plot=sns.heatmap(pd.DataFrame(score).iloc[:-1, :].T, annot=True)
    plt.savefig('./media/plot.png')
    score = pd.DataFrame(score).transpose()
    score.to_csv("classification_report.csv")
    df = pd.read_csv(r"classification_report.csv")
    df.rename(columns={"Unnamed: 0": "Labels"}, inplace=True)
    df.rename(columns={"f1-score": "f1_score"}, inplace=True)
    json_records = df.reset_index().to_json(orient="records")
    data = []
    data = json.loads(json_records)
    context = {"d": data,'f':''}

    return render(request, "table.html", context)
