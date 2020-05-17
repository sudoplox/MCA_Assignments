import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import  warnings
warnings.filterwarnings("ignore")
def helper(i,j):
    #print(" Outerloop: ",i,"Innerloop",j)
    return

def relevance_feedback(vec_docs, vec_queries, sim, n=10):
    beta = 0.4
    rf_sim = sim
    alpha = 0.6
    

    iteration=3
    for i in range(iteration):

        for j in range(vec_queries.shape[0]):
            bottomn = np.argsort(rf_sim[:, j])[:n]  # Get the n most irrelevant documents.
            # print(j)
            helper(i,j)
            for k in bottomn:
                vec_queries[j] =  vec_queries[j] - (beta*vec_docs[k]) # decrease significance
                # print(k)
                helper(j,k)
            topn=np.argsort(-rf_sim[:, j])[:n] # Get the n most relevant documents.
            for k in topn:
                vec_queries[j] = (alpha*vec_docs[k]) + vec_queries[j] # increase significance
                helper(j,k)
                # print(k)
        rf_sim = cosine_similarity(vec_docs, vec_queries)

    return rf_sim  

def relevance_feedback_exp(vec_docs, vec_queries, sim, tfidf_model, n=10):
    beta = 0.2
    number_of_terms = 9
    alpha = 0.8
    
    item=tfidf_model.vocabulary_.items()
    rf_sim = sim
    inv_dic = {v: k for k, v in item}
    for j in range (0, vec_queries.shape[0]):
        new_term_set = []

        bottomn = np.argsort(rf_sim[:, j])[:n]
        for k in bottomn:
            helper(j,k)
            vec_queries[j] =vec_queries[j]- (beta* vec_docs[k])


        topn = np.argsort(-rf_sim[:, j])[:n]
        for k in topn:
            helper(j,k)
            vec_queries[j] = (alpha * vec_docs[k])+vec_queries[j]

        for k in topn:
            helper(j,k)
            top_term_indices= np.argsort(-vec_docs[k,:]) # indices of top terms
            top_term_indices= top_term_indices[:number_of_terms] # indices of top 9

            for term_index in top_term_indices: # indices of top terms iter
                new_term_set.append(inv_dic[term_index])

        vec_queries[j] += tfidf_model.transform(new_term_set)[0][:]

    rf_sim = cosine_similarity(vec_docs, vec_queries)
    
    return rf_sim
