import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import LinearRegression
import numpy as np
from xgboost import XGBClassifier

random.seed(11785085)

#------------------------------------------------------------------------
#Optimal PCA reduction computation
'''
eigenvalues = np.array([3.72115726, 1.36022069, 1.05436192, 0.97277238, 0.93984098, 0.81205422,
 0.71984453, 0.59525019, 0.4579332,  0.25623966, 0.11056942, 0])
explained_variance_ratio = eigenvalues / eigenvalues.sum()

cumulative_explained_variance = np.cumsum(explained_variance_ratio)

print('Cumulative Explained Variance:', cumulative_explained_variance)

plt.figure(figsize=(10, 7))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='--', label='Explained Variance')
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='x', linestyle='-', label='Cumulative Explained Variance')
plt.title('Variance explanined Plot')
plt.xlabel('Number of Principal Components')
plt.ylabel('Variance Explained')
plt.legend()
plt.grid()
plt.show()
'''
#------------------------------------------------------------------------

data = pd.read_csv('musicData.csv')

#------------------------------------------------------------------------
#DATA PROCESSING
#1. I computed the average of all the songs that did not have duration -1 and set missing durations to that value (245503)
#2. I figured I the features were correlated as well, so I used linear regression to fill in the missing tempo data.
data = data.dropna()
data = data.replace({'duration_ms': {-1: 245503}})
data['tempo'] = pd.to_numeric(data['tempo'], errors='coerce')
data_with_tempo = data.dropna(subset=['tempo'])
data_without_tempo = data[data['tempo'].isna()]
predictors = ['acousticness', 'danceability', 'energy', 'loudness']

model = LinearRegression()
model.fit(data_with_tempo[predictors], data_with_tempo['tempo'])

predicted_tempos = model.predict(data_without_tempo[predictors])
data_without_tempo['tempo'] = predicted_tempos

data_filled = pd.concat([data_with_tempo, data_without_tempo])
data_filled['mode'] = data_filled['mode'].apply(lambda x: 1 if x == 'major' else 0)
'''
plt.figure(figsize=(10, 8))
plt.scatter(data_filled['energy'], data_filled['tempo'], alpha=0.5)
plt.title('Energy vs Tempo (Filled with Multiple Predictors)')
plt.xlabel('Energy')
plt.ylabel('Tempo')
plt.show()
'''
print(data_filled.shape)

#end of data processing
#------------------------------------------------------------------------

data = data_filled

features = ['popularity', 'acousticness', 'danceability', 'duration_ms', 'energy',
            'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'mode', 'valence']
target = 'music_genre'

genres = ['Electronic', 'Anime', 'Jazz', 'Alternative', 'Country', 'Rap', 'Blues', 'Rock', 'Classical', 'Hip-Hop']
test_dfs = []
train_dfs = []

for genre in genres:
    genre_data = data[data[target] == genre]
    if len(genre_data) < 5000:
        continue
    test_df = genre_data.sample(n=500, random_state=42) #test set sample
    train_df = genre_data.drop(test_df.index).sample(n=4500, random_state=42) #training set sample
    test_dfs.append(test_df)
    train_dfs.append(train_df)

test_data = pd.concat(test_dfs)
train_data = pd.concat(train_dfs)

X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=9)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
#eigenvalues = pca.explained_variance_
#print(eigenvalues)

xgb_classifier = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, learning_rate=0.1)
xgb_classifier.fit(X_train_pca, y_train_encoded)

y_scores = xgb_classifier.predict_proba(X_test_pca)

y_test_binarized = label_binarize(y_test_encoded, classes=np.arange(len(genres)))
n_classes = y_test_binarized.shape[1]

fpr, tpr, _ = roc_curve(y_test_binarized.ravel(), y_scores.ravel())
roc_auc = auc(fpr, tpr)
print("AUROC:", roc_auc)

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
