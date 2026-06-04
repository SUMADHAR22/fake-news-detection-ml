import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

COLS = ['id','label','statement','subject','speaker','job','state','party',
        'barely_true','false','half_true','mostly_true','pants_fire','context']

def load_data(train_path, test_path):
    train = pd.read_csv(train_path, sep='\t', header=None, names=COLS)
    test  = pd.read_csv(test_path,  sep='\t', header=None, names=COLS)
    return train, test

def binarize(label):
    return 0 if label in ['false', 'pants-fire', 'barely-true'] else 1

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [stemmer.stem(t) for t in text.split() if t not in stop_words]
    return ' '.join(tokens)

def preprocess(train, test):
    for df in [train, test]:
        df['binary_label'] = df['label'].apply(binarize)
        df['clean'] = df['statement'].apply(clean_text)
    return train, test
