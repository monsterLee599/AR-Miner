import numpy as np

class ranking:
    def __init__(self,ds,lda_matrix, time_interval=15, group_weights=[0.85,0,0.15], instance_weights=[0.25,0.25,0.25,0.25]):
        self.ds = ds
        self.lda_matrix = lda_matrix
        self.group_count = len(lda_matrix[0,:])
        self.num_of_reviews = len(self.ds)
        self.interval = time_interval
        self.group_rankings = None
        self.group_score = None
        self.group_weights = group_weights
        self.instance_weights = instance_weights

    def get_group_rankings(self):
        if self.group_rankings is not None:
            return (self.group_rankings, self.group_score)
        volumes = self._group_volume()
        high_volume = max(volumes)
        average_ratings = self._group_average_rating()
        time_series = self._group_time_series()

        group_scores = [ (self.group_weights[0] * float(volumes[idx]/high_volume) + self.group_weights[1]*average_ratings[idx]+ time_series[idx]*self.group_weights[2],idx+1) for idx in range(self.group_count)]
        group_scores.sort(reverse=True)
        scores = np.zeros(self.group_count)
        rankings = np.zeros(self.group_count)
        for i, (x, y) in enumerate(group_scores):
            scores[y - 1] = x
            rankings[y - 1] = i + 1
        self.group_rankings = rankings
        self.group_score = scores
        return (rankings, scores)

    def get_instance_score(self):
        r = self._instance_rating()
        p = self._instance_proportion()
        t = self._instance_time()
        d = self._instance_duplicate()
        r = self._instance_rating()
        p = self._instance_proportion()
        t = self._instance_time()

        return self.instance_weights[0]*r + self.instance_weights[1]*p + self.instance_weights[2]*t + self.instance_weights[3]*d

    def _group_volume(self):
        return (self.lda_matrix.sum(axis=0))

    def _group_average_rating(self):
        ratings = self.ds["rating"].values
        volumes = self._group_volume()
        average_ratings = np.array([(volumes[g] / np.dot(self.lda_matrix[:, g], ratings)) for g in range(self.group_count)])
        return average_ratings

    def _get_time(self):
        times = self.ds["date"].values.copy()
        res = np.zeros(len(times)).astype(int)
        for i in range(len(times)):
            month,day,year = times[i].split("-")
            res[i] = int(month)*31+ int(day)+int(year)*365
        self.ds["timestamp"] = res
        return res

    def _group_time_series(self):
        times = self._get_time()
        t0 = min(times)
        times = (times - t0) // self.interval
        interval_num = max(times)+1
        
        v = np.array([(times==i).sum() for i in range(interval_num)])
        v_g = np.array([self.lda_matrix[times==i,:].sum(axis=0) for i in range(interval_num)])
        p_g = np.array([v_g[i] / v[i] for i in range(interval_num)])
        p = p_g.sum(axis=0)
        l = np.array([i+1 for i in range(interval_num)])
        f = ((p_g.T * l).T / p).sum(axis=0)
        return f

    def _instance_rating(self):
        return 1 / self.ds["rating"].values

    def _instance_time(self):
        times = self._get_time()
        t0 = min(times)
        times = (times - t0) // self.interval
        return times

    def _instance_proportion(self):
        _, score = self.get_group_rankings()
        self.ds["proportion"] = (self.lda_matrix * score).sum(axis=1)
        return self.ds["proportion"].values

    def _instance_duplicate(self,similarity_cutoff=0.8):
        is_duplicate = np.array([False for i in range(self.num_of_reviews)])
        duplicate_num = np.array([0 for i in range(self.num_of_reviews)])
        reviews = self.ds["review"].values
        similarity_cutoff = 0.8

        for i in range(0,self.num_of_reviews):
            if is_duplicate[i] != False:
                continue
            data_count = 1
            for j in range(i+1,self.num_of_reviews):
                if self._jaccard_sim(reviews[i],reviews[j]) >= similarity_cutoff:
                    is_duplicate[j] = True
                    data_count += 1
                    self.ds["rating"].iloc[i] = min(self.ds["rating"].iloc[i],self.ds["rating"].iloc[j])
                    self.ds["timestamp"].iloc[i] = max(self.ds["timestamp"].iloc[i],self.ds["timestamp"].iloc[j])
                    self.ds["proportion"].iloc[i] = max(self.ds["proportion"].iloc[i],self.ds["proportion"].iloc[j])
            duplicate_num[i] = data_count

        self.ds = self.ds[is_duplicate==False]
        self.lda_matrix = self.lda_matrix[is_duplicate==False]
        self.num_of_reviews = len(self.ds)
        max_duplicates = max(duplicate_num)
        return duplicate_num[is_duplicate==False] / max_duplicates

    def _jaccard_sim(self,instance1, instance2):
        count_intersect = 0
        count_1 = len(instance1)
        count_2 = len(instance2)
        i = 0
        j = 0
        while(i<count_1 and j <count_2):
            if instance1[i]==instance2[j]:
                count_intersect = count_intersect+1
                i = i+1
                j = j+1
            elif instance1[i]>instance2[j]:
                j = j+1
            else :
                i = i+1
        return float(count_intersect/(count_1 + count_2 - count_intersect))

if __name__ == "__main__":
    pass