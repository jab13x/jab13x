# Information.
'''This code implements a very rudimentary RNN-based sentence sentiment classifier. Even though this type of RNN is not very good at 
sentiment classification, it sets the foundation for more suitable solutions like LSTMs (also on my GitHub)'''

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
training_set = load_sst_data(sst_home + '/train.txt')
dev_set = load_sst_data(sst_home + '/dev.txt')
test_set = load_sst_data(sst_home + '/test.txt')

print('Training size: {}'.format(len(training_set)))
print('Dev size: {}'.format(len(dev_set)))
print('Test size: {}'.format(len(test_set)))


# Converting data to index vectors and adding padding and unknown tokens.
def sentence_to_padded_index_sequence(datasets):    
    PADDING = "<PAD>"
    UNKNOWN = "<UNK>"
    SEQ_LEN = 20
    
    def tokenize(string):
        return string.lower().split()
    
    word_counter = collections.Counter()
    for example in datasets[0]:
        word_counter.update(tokenize(example['text']))
    
    vocabulary = set([word for word in word_counter if word_counter[word] > 10])
    vocabulary = list(vocabulary)
    vocabulary = [PADDING, UNKNOWN] + vocabulary
        
    word_indices = dict(zip(vocabulary, range(len(vocabulary))))
    indices_to_words = {v: k for k, v in word_indices.items()}
        
    for i, dataset in enumerate(datasets):
        for example in dataset:
            example['index_sequence'] = np.zeros((SEQ_LEN), dtype=np.int32)
            
            token_sequence = tokenize(example['text'])
            padding = SEQ_LEN - len(token_sequence)
            
            for i in range(SEQ_LEN):
                if i >= padding:
                    if token_sequence[i - padding] in word_indices:
                        index = word_indices[token_sequence[i - padding]]
                    else:
                        index = word_indices[UNKNOWN]
                else:
                    index = word_indices[PADDING]
                example['index_sequence'][i] = index
    return indices_to_words, word_indices
    
indices_to_words, word_indices = sentence_to_padded_index_sequence([training_set, dev_set, test_set])


# Defining batch evaluation function.
def evaluate_classifier(classifier, eval_set):
    correct = 0
    hypotheses = classifier(eval_set)
    for i, example in enumerate(eval_set):
        hypothesis = hypotheses[i]
        if hypothesis == example['label']:
            correct += 1        
    return correct / float(len(eval_set))


# Setting up the RNN.
class RNNSentimentClassifier:
    def __init__(self, vocab_size, sequence_length):
        self.learning_rate = 0.2
        self.training_epochs = 500
        self.display_epoch_freq = 5
        self.dim = 24
        self.embedding_dim = 8
        self.batch_size = 256
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.l2_lambda = 0.001
        
        self.trainable_variables = []

        self.E = tf.Variable(tf.random.normal([self.vocab_size, self.embedding_dim], stddev=0.1))
        self.trainable_variables.append(self.E)
        
        self.W_cl = tf.Variable(tf.random.normal([self.dim, 2], stddev=0.1))
        self.b_cl = tf.Variable(tf.random.normal([2], stddev=0.1))
        self.trainable_variables.append(self.W_cl)
        self.trainable_variables.append(self.b_cl)
        
        self.W_rnn = tf.Variable(tf.random.normal([self.embedding_dim+self.dim, self.dim], stddev=0.1))
        self.b_rnn = tf.Variable(tf.random.normal([self.dim], stddev=0.1))
        self.trainable_variables.append(self.W_rnn)
        self.trainable_variables.append(self.b_rnn)

    def model(self,x):
        self.x_slices = tf.split(x, self.sequence_length, 1)
        self.h_zero = tf.zeros([self.batch_size, self.dim])  
        
        def step(x, h_prev):
            embeddings = tf.nn.embedding_lookup(self.E, x)
            embeddings = tf.reshape(embeddings,[self.batch_size, self.embedding_dim])
            concatenation = tf.concat([embeddings, h_prev], 1)
            h = tf.matmul(concatenation,self.W_rnn)+self.b_rnn
            h = tf.tanh(h)
            return h
                
        h_prev = self.h_zero
        for slice in self.x_slices:
          h_prev = step(slice, h_prev)
        
        sentence_representation = h_prev

        logits = tf.matmul(sentence_representation, self.W_cl) + self.b_cl
        return logits

    def train(self, training_data, dev_set):
        def get_minibatch(dataset, start_index, end_index):
            indices = range(start_index, end_index)
            vectors = np.vstack([dataset[i]['index_sequence'] for i in indices])
            labels = [dataset[i]['label'] for i in indices]
            return vectors, labels
      
        print('Training.')

        for epoch in range(self.training_epochs):
            random.shuffle(training_set)
            avg_cost = 0.
            total_batch = int(len(training_set) / self.batch_size)
            
            for i in range(total_batch):
                minibatch_vectors, minibatch_labels = get_minibatch(training_set, self.batch_size * i, self.batch_size * (i + 1))

                with tf.GradientTape() as tape:
                  logits = self.model(minibatch_vectors)
                
                  self.l2_cost = self.l2_lambda * (tf.reduce_sum(tf.square(self.W_rnn)) + tf.reduce_sum(tf.square(self.W_cl)))

                  total_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=minibatch_labels, logits=logits) + self.l2_cost)
        
                optimizer = tf.optimizers.SGD(self.learning_rate)
                gradients = tape.gradient(total_cost, self.trainable_variables)
                gvs = zip(gradients, self.trainable_variables)
                capped_gvs = [(tf.clip_by_norm(grad, 5.0), var) for grad, var in gvs if grad is not None]
                optimizer.apply_gradients(capped_gvs)
                                                                            
                avg_cost += total_cost / total_batch
                
            if (epoch+1) % self.display_epoch_freq == 0:
                tf.print("Epoch:", (epoch+1), "Cost:", avg_cost, \
                    "Dev acc:", evaluate_classifier(self.classify, dev_set[0:256]), \
                    "Train acc:", evaluate_classifier(self.classify, training_set[0:256]))  
    
    def classify(self, examples):
        vectors = np.vstack([example['index_sequence'] for example in examples])
        logits = self.model(vectors)
        return np.argmax(logits, axis=1)
    

# Training and evaluating.
classifier = RNNSentimentClassifier(len(word_indices), 20)
classifier.train(training_set, dev_set)