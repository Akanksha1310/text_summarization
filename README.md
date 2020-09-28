
# Text summarization implementation using the TextRank algorithm

TextRank is a derivative of the famous PageRank algorithm by google. In PageRank, a matrix is generated in which each element represents the probability that a user will move from one webpage to another. In the case of TextRank, a cosine similarity matrix is generated which will picture how similar the sentences are.



```
all_words = [i.split() for i in corpus]
model = Word2Vec(all_words, min_count=1,size= 300)

sen_vector=[]
for i in corpus:
    
    plus=0
    for j in i.split():
        plus+=model.wv[j]
    plus = plus/len(plus)
    
    sen_vector.append(plus)
    
 ```
 
 
For generating word embeddings Word2Vec has been used instead of the Bag-of-Words model or TF-IDF matrix as they only focus on creating a sparse matrix by mapping each word and counting the number of times each word in the vocabulary appears in the document rather than attempting to capture a word's relation with other words. 



```
import networkx as nx

G = nx.from_numpy_array(np.array(sim_mat))  
nx.draw(G, with_labels=True) 
scores = nx.pagerank_numpy(G)
```



After generating the cosine similarity matrix, PageRank ranking algorithm is applied to calculate scores for each sentence and top K sentences with maximum scores form the summary, where K represents the number of desired sentences in the summary.
