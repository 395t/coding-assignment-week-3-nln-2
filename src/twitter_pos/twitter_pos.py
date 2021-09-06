import numpy as np
import pickle
import sys
import io
import os


from helpers import embeddings_to_dict, data_to_mat, word_list_to_embedding, get_act_fn
from net import Net


window_size = 1

# note that we encode the tags with numbers for later convenience
tag_to_number = {
    u'N': 0, u'O': 1, u'S': 2, u'^': 3, u'Z': 4, u'L': 5, u'M': 6,
    u'V': 7, u'A': 8, u'R': 9, u'!': 10, u'D': 11, u'P': 12, u'&': 13, u'T': 14,
    u'X': 15, u'Y': 16, u'#': 17, u'@': 18, u'~': 19, u'U': 20, u'E': 21, u'$': 22,
    u',': 23, u'G': 24
}

embeddings = embeddings_to_dict('./data/Tweets/embeddings-twitter.txt')
vocab = embeddings.keys()

# we replace <s> with </s> since it has no embedding, and </s> is a better embedding than UNK
xt, yt = data_to_mat('./data/Tweets/tweets-train.txt', vocab, tag_to_number, window_size=window_size,
                     start_symbol=u'</s>')
xdev, ydev = data_to_mat('./data/Tweets/tweets-dev.txt', vocab, tag_to_number, window_size=window_size,
                         start_symbol=u'</s>')
xdtest, ydtest = data_to_mat('./data/Tweets/tweets-devtest.txt', vocab, tag_to_number, window_size=window_size,
                             start_symbol=u'</s>')

data = {
    'x_train': xt, 'y_train': yt,
    'x_dev': xdev, 'y_dev': ydev,
    'x_test': xdtest, 'y_test': ydtest
}



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset


num_epochs = 50
num_tags = 25
hidden_size = 256
batch_size = 16
embedding_dimension = 50
example_size = (2 * window_size + 1) * embedding_dimension
num_examples = data['y_train'].shape[0]
num_batches = num_examples // batch_size
save_every = num_batches // 5 # save training information 5 times per epoch


nonlinearity_list = ['relu', 'prelu', 'elu', 'silu', 'mish', 'gelu']    
learning_rate = 1e-2 # 1e-3 # 1e-4 # 1e-5


for nonlinearity_name in nonlinearity_list:
    
    for n_exp in range(3):


        print('>> %s #%d' % (nonlinearity_name, n_exp))

        history = {
                "lr": learning_rate,
                'train_loss': [], 'val_loss': [], 'test_loss': [],
                'train_acc': [], 'val_acc': [], 'test_acc': []
            }
          
        act_fn = get_act_fn(nonlinearity_name)
        net = Net(act_fn, example_size, hidden_size, num_tags)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
              
        for epoch in range(num_epochs):
            
            print('>>>> Epoch', epoch + 1)            

            # shuffle data every epoch
            indices = np.arange(num_examples)
            np.random.shuffle(indices)
            data['x_train'] = data['x_train'][indices]
            data['y_train'] = data['y_train'][indices]  

            # reset loss            
            running_loss = 0.0
  
            # train
            net.train()

            for i in range(num_batches):
                offset = i * batch_size
                inputs = word_list_to_embedding(data['x_train'][offset:offset + batch_size, :],
                                            embeddings, embedding_dimension)
                labels = data['y_train'][offset:offset + batch_size] 

                inputs = torch.tensor(inputs)
                labels = torch.tensor(labels).long()
            
                # training
                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss

            # training loss
            train_loss = running_loss / len(data['y_train'])
            print('>>>>>> [train_loss] \t %.4f' % train_loss)


            # validation
            net.eval()

            with torch.no_grad():
              inputs = word_list_to_embedding(data['x_train'], embeddings, embedding_dimension)
              labels = data['y_train']

              inputs = torch.tensor(inputs)
              labels = torch.tensor(labels).long()

              outputs = net(inputs)
              
              # compute loss
              train_loss = criterion(outputs, labels)

              # compute accuracy
              predictions = torch.argmax(outputs, dim=1)
              train_acc = (predictions == labels).float().sum() / len(outputs)

              # print statistics
              print('>>>>>> [train_loss] \t %.4f' % train_loss)
              print('>>>>>> [train_acc] \t %.4f' % train_acc)
              history["train_loss"].append(train_loss)
              history["train_acc"].append(train_acc)

            with torch.no_grad():
              inputs = word_list_to_embedding(data['x_dev'], embeddings, embedding_dimension)
              labels = data['y_dev']

              inputs = torch.tensor(inputs)
              labels = torch.tensor(labels).long()

              outputs = net(inputs)
              
              # compute loss
              val_loss = criterion(outputs, labels)

              # compute accuracy
              predictions = torch.argmax(outputs, dim=1)
              val_acc = (predictions == labels).float().sum() / len(outputs)

              # print statistics
              print('>>>>>> [val_loss] \t %.4f' % val_loss)
              print('>>>>>> [val_acc] \t %.4f' % val_acc)
              history["val_loss"].append(val_loss)
              history["val_acc"].append(val_acc)

            # testing
            with torch.no_grad():
              inputs = word_list_to_embedding(data['x_test'], embeddings, embedding_dimension)
              labels = data['y_test']

              inputs = torch.tensor(inputs)
              labels = torch.tensor(labels).long()

              outputs = net(inputs)
              
              # compute loss
              test_loss = criterion(outputs, labels)

              # compute accuracy
              predictions = torch.argmax(outputs, dim=1)
              test_acc = (predictions == labels).float().sum() / len(outputs)

              # print statistics
              print('>>>>>> [test_loss] \t %.4f' % test_loss)
              print('>>>>>> [test_acc] \t %.4f' % test_acc)
              history["test_loss"].append(test_loss)
              history["test_acc"].append(test_acc)

              
        pickle.dump(history, open("./twitter_pos_" + nonlinearity_name + '_exp%d' % n_exp + ".p", "wb"))