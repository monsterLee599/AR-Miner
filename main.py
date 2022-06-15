from data import dataset
from LDA import LDA
from ranking import ranking

if __name__ == "__main__":
    ds = dataset("./Raw/Facebook").get_pd()
    lda = LDA(ds.values[:,-1])
    pr = lda.run()
    rk = ranking(ds,pr)
    print("Group Rank: ",rk.get_group_rankings())
    print("Instance Rank",rk.get_instance_score())
    