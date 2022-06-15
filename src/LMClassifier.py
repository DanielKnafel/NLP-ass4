from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

classifier = pipeline('sentiment-analysis')
classifier.fit(['I love this movie!', 'This is a bad movie.'])

