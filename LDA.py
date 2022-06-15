from data import dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk


class LDA:
    def __init__(self, reviews):
        cntVector = CountVectorizer(max_df=0.85)
        self.cntTF = cntVector.fit_transform(reviews)

    def run(self):
        print("[Info] Begin run LatentDirichletAllocation")
        lda_solver = LatentDirichletAllocation(n_components=8, max_iter=10)
        pr = lda_solver.fit_transform(self.cntTF)
        print("[Info] Finish LatentDirichletAllocation")
        return pr

# if __name__=="__main__":
#     ds = dataset("./Raw/Facebook")
#     lda = LDA(ds.get_array()[:,-1][:1000])
#     lda.run()