import networkx as nx
import csv
import os,glob
import pickle
import sklearn
from networkx.algorithms import bipartite
import scipy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from sklearn.utils.extmath import safe_sparse_dot
from sklearn.metrics.pairwise import check_pairwise_arrays
from random import shuffle
from sklearn.preprocessing import normalize


# movie_id: 1 - 17770
# user_id: 1 - 2649429
# B[movie_id][user_id+17770] = weight
# A_d[movie_id-1][user_id-1] = weight
# sim[movie_id-1][movie_id-1]


def predict(movie_i, user,  sim, A_d): 
    movie_list = []
    for movie in range(0, 17770): 
        if A_d[movie][user-1] != 0: 
            movie_list.append(movie)
    sum_btm = 0
    sum_top = 0
    for movie_j in movie_list: 
        sum_btm = sum_btm + sim[movie_i-1][movie_j]
        sum_top = sum_top + sim[movie_i-1][movie_j] * A_d[movie_j][user-1]
    if sum_btm != 0:
        predict = sum_top / sum_btm
    else:
        predict = None
    return predict




def adj_cosine_similarity(X, Y=None, dense_output=True):
    
    X, Y = check_pairwise_arrays(X, Y)
    X_normalized = normalize(X, axis=0, copy=True)
    if X is Y:
        Y_normalized = X_normalized
    else:
        Y_normalized = normalize(Y, copy=True)

    K = safe_sparse_dot(X_normalized, Y_normalized.T,
                        dense_output=dense_output)

    return K



training_file = ['01', '02']
test_file = '00'
TEST_NUM = 500

def main():
    print('Training file: ' + ' '.join(training_file))
    print('Testing file: ' + str(test_file))
    print('Num of tests: '+ str(TEST_NUM))

    print("Test started.")
    print('Loading tests...')
    G_test = nx.read_gpickle('./correct-data/' +test_file+'-graph')
    test_list = list(G_test.edges(data='weight'))


    print('Creating graph...')
    G = nx.Graph()
    G.add_nodes_from(range(1, 17771), bipartite=0)
    G.add_nodes_from(range(17771, (2649429+17770+1)), bipartite=1)
    
    for file in training_file: 
        print('Processing '+ file)
        G_new = nx.read_gpickle('./correct-data/'+file+'-graph')
        G.add_edges_from(G_new.edges(data=True))


    print("Creating sparse matrix...")
    A = bipartite.matrix.biadjacency_matrix(G, row_order = range(1, 17771))
    print("Creating dense matrix...")
    A_d = A.toarray()
    print("Similarity matrices...")
    sim_pearson = cosine_similarity(A)
    sim_adjcos = adj_cosine_similarity(A)

    print('Begin testing!')
    #(movie_id, user_ID+17770, weight)
    shuffle(test_list)

    
    test_num = 0
    for test in test_list:
        test_movie_id, test_user_id, test_weight = test[0], (test[1]-17770), test[2]
        pearson_predict = predict(test_movie_id, test_user_id, sim_pearson, A_d)
        adj_predict = predict(test_movie_id, test_user_id, sim_adjcos, A_d)
        if pearson_predict != None and adj_predict != None:
            with open('./results' + ''.join(training_file)+'-'+ ''.join(test_file)+'.txt', 'a') as f:
                f.write(str(pearson_predict)+','+str(adj_predict)+','+ str(test_weight)+'\n')
                test_num += 1
        if test_num >= TEST_NUM:
            break

    

    pickle.dump(results, open('./results', 'wb'))
    print('results file saved')

main()





