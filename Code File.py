import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
from wordcloud import WordCloud
import re
import string
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_auc_score

df = pd.read_csv('datasets/tweets_dataset.csv', lineterminator='\n')

# data info
df.info()
df.head()


#DATASET PREPROCESSING

  # remove usernames, links, special characters, numbers and punctuations
def text_cliner(tweet):
    tweet = re.sub('@[^\s]+','',str(tweet))
    tweet = re.sub('http[^\s]+','',tweet)
    tweet = re.sub('#','',tweet)
    tweet = re.sub('&amp;','',tweet)
    return tweet
df['clean_tweet'] = df['tweet'].apply(text_cliner)

# removing special characters
df['clean_tweet'] = df['clean_tweet'].str.replace("[^a-zA-Z0-9#]", " ")

# remove short words and applying lower case
df['clean_tweet'] = df['clean_tweet'].apply(lambda x: " ".join([w.lower() for w in str(x).split() if len(w)>3]))


df.head()



#removing stop words
stop_words = stopwords.words('english')
df['clean_tweet'] = df['clean_tweet'].apply(lambda x: ' '.join([word for word in x.lower().split() if word not in (stop_words)]))
df



#Tokenization

tokenized_tweet = df['clean_tweet'].apply(lambda x: x.split())
tokenized_tweet.head()

# lemmatization

lemmatizer = WordNetLemmatizer()
tokenized_tweet = tokenized_tweet.apply(lambda sentence: [lemmatizer.lemmatize(word) for word in sentence])
tokenized_tweet.head()


# combine words into single sentence
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = " ".join(tokenized_tweet[i])
    
df['clean_tweet'] = tokenized_tweet
df.head()


#drop row's containing NAN values
df['clean_tweet'] = df['clean_tweet'].dropna()
df


# visualize the frequent words
all_words = " ".join([sentence for sentence in df['clean_tweet']])

wordcloud = WordCloud(width=800, height=600, max_font_size=100, colormap="Blues").generate(all_words)

# plot the graph
plt.figure(figsize=(15,8), facecolor = 'k')
plt.imshow(wordcloud) #, interpolation='bilinear')
plt.axis('off')
plt.show()


# Strip unwanted characters from all column names
df.columns = df.columns.str.strip()
 
# Confirm the updated column names
print(df.columns)


df = df[df['clean_tweet'].notnull() & (df['clean_tweet'] != "")]


df.columns = df.columns.str.strip()  # Strip unwanted characters

df['label'] = df['label'].str.strip()
df = df[df['label'] != ""]


df['label'] = df['label'].astype(int)

# frequent words visualization for happy
all_words = " ".join([sentence for sentence in df['clean_tweet'][df['label']==0]])
 
wordcloud = WordCloud(width=800, height=600, max_font_size=100, colormap="Blues").generate(all_words)
 
# plot the graph
plt.figure(figsize=(15,8), facecolor = 'k')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# frequent words visualization for depressed
all_words = " ".join([sentence for sentence in df['clean_tweet'][df['label']==1]])

wordcloud = WordCloud(width=800, height=600, max_font_size=100, colormap="Blues").generate(all_words)

# plot the graph
plt.figure(figsize=(15,8), facecolor = 'k')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Vectoriser: Feature Extraction
vectorizer = CountVectorizer(stop_words='english') 
bow = vectorizer.fit_transform(df['tweet'])  # Transform your text data into a feature matrix
 
# Now you can split your data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(bow, df['label'], random_state=42, test_size=0.20)

# Dictionary to store model names and their accuracies
accuracy_scores = {}


## LOGISTIC REGRESSIOn
from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression()
# training the algorithm 
log_model.fit(x_train, y_train)
y_pred = log_model.predict(x_test)

print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))
print('F1 Score: %.3f' % f1_score(y_test, y_pred))
print('Accuracy: %.3f' %accuracy_score(y_test,y_pred))
accuracy_scores['Logistic Regression'] =accuracy_score(y_test,y_pred)

print(classification_report(y_test, y_pred))

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix)

cm_display.plot()
plt.show()

log_scores = log_model.predict_proba(x_test)[:, 1]  # Logistic Regression
fpr_log, tpr_log, _ = roc_curve(y_test, log_scores)
auc_log = auc(fpr_log, tpr_log)

##Naive Bayes

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn import metrics

nb_model = MultinomialNB()
nb_model.fit(x_train, y_train)
y_pred = nb_model.predict(x_test)

print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))
print('F1 Score: %.3f' % f1_score(y_test, y_pred))
print('Accuracy: %.3f' %accuracy_score(y_test,y_pred))
accuracy_scores['Naive Bayes'] = accuracy_score(y_test,y_pred)

print(classification_report(y_test, y_pred))
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix)

cm_display.plot()
plt.show()

nb_scores = nb_model.predict_proba(x_test)[:, 1]  # Naive Bayes
fpr_nb, tpr_nb, _ = roc_curve(y_test, nb_scores)
auc_nb = auc(fpr_nb, tpr_nb)

#K-Nearest Neighbor(KNN) Algorithm
from sklearn.neighbors import KNeighborsClassifier

knn_model= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )
# training the algorithem 
knn_model.fit(x_train, y_train)
y_pred = knn_model.predict(x_test)

print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))
print('F1 Score: %.3f' % f1_score(y_test, y_pred))
print('Accuracy: %.3f' %accuracy_score(y_test,y_pred))


accuracy_scores['KNN'] =accuracy_score(y_test,y_pred)
print(classification_report(y_test, y_pred))
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix)

cm_display.plot()
plt.show()

knn_scores = knn_model.predict_proba(x_test)[:, 1]  # KNN
fpr_knn, tpr_knn, _ = roc_curve(y_test, knn_scores)
auc_knn = auc(fpr_knn, tpr_knn)


#Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(criterion="entropy", random_state=0)
# training the algorithem 
dt_model.fit(x_train, y_train)
# testing
y_pred = dt_model.predict(x_test)

print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))
print('F1 Score: %.3f' % f1_score(y_test, y_pred))
print('Accuracy: %.3f' %accuracy_score(y_test,y_pred))

accuracy_scores['Decision Tree'] = accuracy_score(y_test,y_pred)
print(classification_report(y_test, y_pred))
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix)

cm_display.plot()
plt.show()
dt_scores = dt_model.predict_proba(x_test)[:, 1]  # Decision Tree
fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_scores)
auc_dt = auc(fpr_dt, tpr_dt)



##RANDOM FOREST CLASSIFIER
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# Select features using RandomForest
selector = SelectFromModel(RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42))
selector.fit(x_train, y_train)
# Transform the training and testing data
x_train_reduced = selector.transform(x_train)
x_test_reduced = selector.transform(x_test)
# Initialize your classifier (e.g., Random Forest)
rf_model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
# Train the model using the reduced features
rf_model.fit(x_train_reduced, y_train)
# Predict on the test set
y_pred = rf_model.predict(x_test_reduced)

print("Accuracy:", accuracy_score(y_test, y_pred))
print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))
print('F1 Score: %.3f' % f1_score(y_test, y_pred))
print('Accuracy: %.3f' %accuracy_score(y_test,y_pred))

accuracy_scores['Random Forest'] = accuracy_score(y_test,y_pred)
 
print(classification_report(y_test, y_pred))

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix)
 
cm_display.plot()
plt.show()
rf_scores = rf_model.predict_proba(x_test_reduced)[:, 1]  # Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_scores)
auc_rf = auc(fpr_rf, tpr_rf)



#SUPPORT VECTOR MACHINE
from sklearn.svm import LinearSVC  # Linear SVM (faster)

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

svm_model = LinearSVC(random_state=42, max_iter=1000, tol=0.01)  # Adjust tolerance to speed up convergence
 
# Train the model using the scaled data
svm_model.fit(x_train, y_train)
 
# Predict on the test set
y_pred = svm_model.predict(x_test)
 
# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print('Precision: %.3f' % precision_score(y_test, y_pred, average='weighted'))  # Use weighted for multi-class
print('Recall: %.3f' % recall_score(y_test, y_pred, average='weighted'))  # Use weighted for multi-class
print('F1 Score: %.3f' % f1_score(y_test, y_pred, average='weighted'))  # Use weighted for multi-class
print(classification_report(y_test, y_pred))

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix)

cm_display.plot()
plt.show()

accuracy_scores['SVM'] = accuracy_score(y_test,y_pred)

svm_scores = svm_model.decision_function(x_test)  # SVM (use decision_function for ROC)
fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_scores)
auc_svm = auc(fpr_svm, tpr_svm)


## roc curve
# Plot the combined ROC curve
plt.figure(figsize=(12, 8))
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.2f})', color='blue')
plt.plot(fpr_log, tpr_log, label=f'Logistic Regression (AUC = {auc_log:.2f})', color='green')
plt.plot(fpr_nb, tpr_nb, label=f'Naive Bayes (AUC = {auc_nb:.2f})', color='red')
plt.plot(fpr_knn, tpr_knn, label=f'KNN (AUC = {auc_knn:.2f})', color='purple')
plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {auc_dt:.2f})', color='orange')
plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {auc_svm:.2f})', color='cyan')

# Plot random chance line
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance', alpha=0.7)

# Customize plot
plt.title('Combined ROC Curve', fontsize=16)
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.legend(loc='lower right', fontsize=12)
plt.grid(alpha=0.4)
plt.show()


##ACCURACY COMPARISION
# Plotting the histogram
plt.figure(figsize=(10, 6))
model_names = list(accuracy_scores.keys())
accuracy_values = list(accuracy_scores.values())
 
plt.bar(model_names, accuracy_values, color=['blue', 'green', 'orange', 'purple', 'red', 'cyan'])
plt.xlabel('Models', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.title('Model Accuracy Comparison', fontsize=16)
plt.ylim(0, 1)  # Accuracy ranges from 0 to 1
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
 
# Annotating the bars with accuracy values
for i, val in enumerate(accuracy_values):
    plt.text(i, val + 0.02, f'{val:.2f}', ha='center', fontsize=12)
 
plt.tight_layout()
plt.show()


