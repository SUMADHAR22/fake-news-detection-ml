from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack, csr_matrix

NUM_COLS = ['barely_true', 'false', 'half_true', 'mostly_true', 'pants_fire']

def build_features(train, test):
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train = tfidf.fit_transform(train['clean'])
    X_test  = tfidf.transform(test['clean'])

    num_train = csr_matrix(train[NUM_COLS].fillna(0).values)
    num_test  = csr_matrix(test[NUM_COLS].fillna(0).values)

    X_train = hstack([X_train, num_train])
    X_test  = hstack([X_test,  num_test])

    return X_train, X_test, tfidf

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_train, y_train)
    return model
