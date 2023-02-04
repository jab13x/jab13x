# Information.
'''This code implements a simple MLP sentence sentiment classifier with two ReLu hidden layers, L2 regularization, 
and a 20% drop rate to avoid overfitting.'''

import re
import random
import collections
import numpy as np
import tensorflow as tf


# Loading data and, in this case, setting a 2-way positive/negative classification instead of 5-way.
def load_sst_data(path, easy_label_map={0:0, 1:0, 2:None, 3:1, 4:1}):
    data = []
    with open(path) as f:
        for i, line in enumerate(f): 
            example = {}
            example['label'] = easy_label_map[int(line[1])]
            if example['label'] is None:
                continue

            text = re.sub(r'\s*(\(\d)|(\))\s*', '', line)
            example['text'] = text[1:]
            data.append(example)

    random.seed(1)
    random.shuffle(data)
    return data

sst_home = '###Change this to set the path for the data.###' # Setting home of data.
training_set = load_sst_data(sst_home + 'train.txt')
dev_set = load_sst_data(sst_home + 'dev.txt')
test_set = load_sst_data(sst_home + 'test.txt')

print('Training size: {}'.format(len(training_set)))
print('Dev size: {}'.format(len(dev_set)))
print('Test size: {}'.format(len(test_set)))


# Extracting BoW feature vectors.
def feature_function(datasets):
    
    def tokenize(string):
        return string.split()
    
    word_counter = collections.Counter()
    for example in datasets[0]:
        word_counter.update(tokenize(example['text']))
    
    vocabulary = set([word for word in word_counter if word_counter[word] > 10])
                                
    feature_names = set()
    for i, dataset in enumerate(datasets):
        for example in dataset:
            example['features'] = collections.defaultdict(float)
            
            word_counter = collections.Counter(tokenize(example['text']))
            for x in word_counter.items():
                if x[0] in vocabulary:
                    example["features"]["word_count_for_" + x[0]] = x[1]
            
            feature_names.update(example['features'].keys())
                            
    feature_indices = dict(zip(feature_names, range(len(feature_names))))
    indices_to_features = {v: k for k, v in feature_indices.items()}
    dim = len(feature_indices)
                
    for dataset in datasets:
        for example in dataset:
            example['vector'] = np.zeros((dim))
            for feature in example['features']:
                example['vector'][feature_indices[feature]] = example['features'][feature]
    return indices_to_features, dim
    
indices_to_features, dim = feature_function([training_set, dev_set, test_set])

print('Vocabulary size: {}'.format(dim))


# Defining batch evaluation function.
def evaluate_classifier(classifier, eval_set):
    correct = 0
    hypotheses = classifier(eval_set)
    for i, example in enumerate(eval_set):
        hypothesis = hypotheses[i]
        if hypothesis == example['label']:
            correct += 1        
    return correct / float(len(eval_set))


# Preparing a logistic regression classifier with two ReLu hidden layers, L2 regularization, and a 20% drop rate to avoid overfitting.
class logistic_regression_classifier:
    def __init__(self, dim, l2=0.1, rate=0.2): 
        self.learning_rate = 0.3 
        self.training_epochs = 100
        self.display_epoch_freq = 1
        self.dim = dim
        self.batch_size = 256

        self.hidden_layer_sizes = [50, 50]
        self.rate = rate

        self.trainable_variables = []

        self.W0 = tf.Variable(tf.random.normal([self.dim,50], stddev=0.1))
        self.b0 = tf.Variable(tf.random.normal([50], stddev=0.1))
        self.trainable_variables.append(self.W0)
        self.trainable_variables.append(self.b0)

        self.W1 = tf.Variable(tf.random.normal([50,50], stddev=0.1))
        self.b1 = tf.Variable(tf.random.normal([50], stddev=0.1))
        self.trainable_variables.append(self.W1)
        self.trainable_variables.append(self.b1)

        self.W2 = tf.Variable(tf.random.normal([50, 2], stddev=0.1), dtype='float32')
        self.b2 = tf.Variable(tf.random.normal([2], stddev=0.1), dtype='float32')
        self.trainable_variables.append(self.W2)
        self.trainable_variables.append(self.b2)

        self.l2_reg = tf.keras.regularizers.L2(l2)

    def model(self,x,rate):
        h_0 = tf.nn.relu(tf.matmul(x, self.W0) + self.b0)
        h_0 = tf.nn.dropout(h_0, self.rate)
        h_1 = tf.nn.relu(tf.matmul(h_0, self.W1) + self.b1)
        h_1 = tf.nn.dropout(h_1, self.rate)
        logits = tf.matmul(h_1, self.W2) + self.b2
        return logits
     

    def train(self, training_set, dev_set):
        def get_minibatch(dataset, start_index, end_index):
            indices = range(start_index, end_index)
            vectors = np.float32(np.vstack([dataset[i]['vector'] for i in indices]))
            labels = [dataset[i]['label'] for i in indices]
            return vectors, labels
      
        print ('Training.')

        for epoch in range(self.training_epochs):
            random.shuffle(training_set)
            avg_cost = 0.
            total_batch = int(len(training_set) / self.batch_size)
            for i in range(total_batch):
                minibatch_vectors, minibatch_labels = get_minibatch(training_set, self.batch_size * i, self.batch_size * (i + 1))
                with tf.GradientTape() as tape:
                  logits = self.model(minibatch_vectors, self.rate)
                  cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=minibatch_labels))
                  cost = self.l2_reg(cost)


                gradients = tape.gradient(cost, self.trainable_variables)
                optimizer = tf.optimizers.SGD(self.learning_rate)
                optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                avg_cost += cost / total_batch
                
            if (epoch+1) % self.display_epoch_freq == 0:
                tf.print ("Epoch:", (epoch+1), "Cost:", avg_cost, \
                    "Dev acc:", evaluate_classifier(self.classify, dev_set[0:500]), \
                    "Train acc:", evaluate_classifier(self.classify, training_set[0:500]))
    
    def classify(self, examples):
        vectors = np.float32(np.vstack([example['vector'] for example in examples]))
        ''' ## Part 3: Turning back on all neurons for test. ## '''
        logits = self.model(vectors, rate=0.0)
        return np.argmax(logits, axis=1)
    

# Training and evaluating.
classifier = logistic_regression_classifier(dim)
classifier.train(training_set, dev_set)

evaluate_classifier(classifier.classify, test_set)
