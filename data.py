import pandas as pd
import numpy
import os
import sys

class dataset:

    def __init__(self,dirname):
        self.reviews = []
        self.review_info = []
        for root, dirs,files in os.walk(dirname+"/review"):
            for file in files:
                file_path=os.path.join(root,file)
                f = open(dirname+"/review/"+file)
                lines = f.readlines()
                self.reviews += [pd.DataFrame(lines, columns=["review"])]
        
        for root, dirs,files in os.walk(dirname+"/appinfo"):
            for file in files:
                file_path=os.path.join(root,file)
                self.review_info += [pd.read_table(file_path, delimiter=' ', index_col=False, header=0, names=['rating','date', 'time', 'id', 'app_version'])]

        for i in range(len(self.reviews)):
            self.review_info[i] = self.review_info[i][:len(self.reviews[i])]
        self.review_info = pd.concat(self.review_info)
        self.reviews = pd.concat(self.reviews)
        self.review_info["review"] = self.reviews["review"]
        self.reviews = self.review_info
        self.reviews.reset_index(drop=True, inplace=True)
        print("[Info] Get %d reviews from %s"%(len(self.reviews),dirname))

    def get_array(self):
        return self.reviews.values

    def get_pd(self):
        return self.reviews

    def _word_tokenize(self,reviews):
        temp = []
        for i in range(len(reviews)):
            tokens = nltk.word_tokenize(reviews[i])
            temp.append(" ".join(tokens))
        return temp




if __name__ == "__main__":
    dataset("./Raw/Facebook")