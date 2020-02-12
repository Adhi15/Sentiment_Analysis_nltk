import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords

def create_word_feat(words):
    useful_words = [word for word in words if word not in stopwords.words("english")]
    dict_words = dict([(word, True) for word in useful_words])
    return dict_words

#create_word_features(["the", "quick", "brown", "quick", "a", "fox"])

neg_reviews = []
for fileid in movie_reviews.fileids('neg'):
       	words = movie_reviews.words(fileid)
        neg_reviews.append((create_word_feat(words), "negative"))

print(neg_reviews[0])    
#print(len(neg_reviews))
 
pos_reviews = []
for fileid in movie_reviews.fileids('pos'):
    words = movie_reviews.words(fileid)
    pos_reviews.append((create_word_feat(words), "positive"))
    
#print(pos_reviews[0])    
#print(len(pos_reviews))
 
train_set = neg_reviews[:550] + pos_reviews[:550]
test_set =  neg_reviews[550:] + pos_reviews[550:]
print(len(train_set),  len(test_set))


classifier = NaiveBayesClassifier.train(train_set)

accuracy = nltk.classify.util.accuracy(classifier, test_set)
print(accuracy)