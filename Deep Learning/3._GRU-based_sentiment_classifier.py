# Information.
'''This code implements a simple GRU RNN sentiment classifier. Some code to plot a simple learning curve is also included.'''

import re
import random
import collections
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


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
        self.learning_rate = 1.0
        self.training_epochs = 500
        self.display_epoch_freq = 5
        self.dim = 12
        self.embedding_dim = 8
        self.batch_size = 256
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.l2_lambda = 0.001
        
        self.trainable_variables = []

        self.E = tf.Variable(tf.random.normal([self.vocab_size, self.embedding_dim], stddev=0.1))
        self.trainable_variables.append(self.E)

        self.W_rnn = tf.Variable(tf.random.normal([self.embedding_dim + self.dim, self.dim], stddev=0.1))
        self.b_rnn = tf.Variable(tf.random.normal([self.dim], stddev=0.1))
        self.trainable_variables.append(self.W_rnn)
        self.trainable_variables.append(self.b_rnn)
        
        self.W_cl = tf.Variable(tf.random.normal([self.dim, 2], stddev=0.1))
        self.b_cl = tf.Variable(tf.random.normal([2], stddev=0.1))
        self.trainable_variables.append(self.W_cl)
        self.trainable_variables.append(self.b_cl)
        
        self.W_z = tf.Variable(tf.random.normal([self.embedding_dim+self.dim, self.dim], stddev=0.1))
        self.b_z = tf.Variable(tf.random.normal([self.dim], stddev=0.1))
        self.trainable_variables.append(self.W_z)
        self.trainable_variables.append(self.b_z)

        self.W_r = tf.Variable(tf.random.normal([self.embedding_dim+self.dim, self.dim], stddev=0.1))
        self.b_r = tf.Variable(tf.random.normal([self.dim], stddev=0.1))
        self.trainable_variables.append(self.W_r)
        self.trainable_variables.append(self.b_r)

                
    def model(self,x):
        self.x_slices = tf.split(x, self.sequence_length, 1)
        self.h_zero = tf.zeros([self.batch_size, self.dim])

        def step(x, h_prev):
            embeddings = tf.nn.embedding_lookup(self.E, x)
            emb_h_prev = tf.concat([embeddings, h_prev], 1)
            z_t = tf.math.sigmoid(tf.matmul(emb_h_prev, self.W_z) + self.b_z)
            r_t = tf.math.sigmoid(tf.matmul(emb_h_prev, self.W_r) + self.b_r)
            emb_h_prev_r_t = tf.concat([embeddings, r_t * h_prev], 1)
            h_tilde = tf.tanh(tf.matmul(emb_h_prev_r_t, self.W_rnn) + self.b_rnn)
            h_t = (1 - z_t) * h_prev + z_t * h_tilde
            return h_t
                
        h_prev = self.h_zero
        
        for t in range(self.sequence_length):
            x_t = tf.reshape(self.x_slices[t], [-1])
            h_prev = step(x_t, h_prev)
        
        logits = tf.matmul(h_prev, self.W_cl) + self.b_cl
        return logits
        
    def train(self, training_data, dev_set):
        def get_minibatch(dataset, start_index, end_index):
            indices = range(start_index, end_index)
            vectors = np.vstack([dataset[i]['index_sequence'] for i in indices])
            labels = [dataset[i]['label'] for i in indices]
            return vectors, labels
    
        print ('Training.')

        train_acc = []
        dev_acc = []
        epochs = []
        for epoch in range(self.training_epochs):
            random.shuffle(training_set)
            avg_cost = 0.
            total_batch = int(len(training_set) / self.batch_size)
            
            for i in range(total_batch):
                minibatch_vectors, minibatch_labels = get_minibatch(training_set, 
                                                                    self.batch_size * i, 
                                                                    self.batch_size * (i + 1))

                with tf.GradientTape() as tape:
                  logits = self.model(minibatch_vectors)
                  l2_cost = self.l2_lambda * (tf.reduce_sum(tf.square(self.W_rnn)) +
                                                   tf.reduce_sum(tf.square(self.W_cl)) + 
                                                   tf.reduce_sum(tf.square(self.W_z)) + 
                                                   tf.reduce_sum(tf.square(self.W_r)))
        
                  total_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=minibatch_labels, logits=logits) + l2_cost)
        
                optimizer = tf.optimizers.SGD(self.learning_rate)
                gradients = tape.gradient(total_cost, self.trainable_variables)
                gvs = zip(gradients, self.trainable_variables)
                capped_gvs = [(tf.clip_by_norm(grad, 5.0), var) for grad, var in gvs if grad is not None]
                optimizer.apply_gradients(capped_gvs)
                                  
                avg_cost += total_cost / total_batch
                
            if (epoch+1) % self.display_epoch_freq == 0:
                dev_acc.append(evaluate_classifier(self.classify, dev_set[0:256]))
                train_acc.append(evaluate_classifier(self.classify, training_set[0:256]))
                epochs.append(epoch+1)
                tf.print("Epoch:", (epoch+1), "Cost:", avg_cost, \
                    "Dev acc:", dev_acc[-1], \
                    "Train acc:", train_acc[-1])  
        return train_acc, dev_acc, epochs
    
    def classify(self, examples):
        vectors = np.vstack([example['index_sequence'] for example in examples])
        logits = self.model(vectors)
        return np.argmax(logits, axis=1)

np.random.seed(1)
tf.random.set_seed(1)

classifier = RNNSentimentClassifier(len(word_indices), 20)


# Training and evaluating.
train_acc, dev_acc, epochs = classifier.train(training_set, dev_set)


# Plotting the test and training learning curves.
def plot_learning_curve(par_values, train_scores, dev_scores, title="Learning Curve", xlab="", ylab="Accuracy", ylim=None):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    
    plt.grid()
    plt.plot(par_values, train_scores, color="r",label="Training score")
    plt.plot(par_values, dev_scores, color="g", label="Dev score")

    plt.legend(loc="best")
    return plt

plt = plot_learning_curve(epochs, train_acc, dev_acc, xlab="Epoch")
plt.show()