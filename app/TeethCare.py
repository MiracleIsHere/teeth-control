from collections import defaultdict, Counter

# class to manage information about recent teeth area
class TeethCare:
    def __init__(self, square_dict, queue_size, critical_loss):
        self.square_ref = square_dict
        self.__square_diff__ = defaultdict(list)
        self.history = defaultdict(list)
        self.history_coords = defaultdict(list)
        self.queue_size = queue_size
        self.critical_loss = critical_loss
        self.imgs = defaultdict(list)

    def addReview(self, name, square, position, img):
        if self.queueIsFull(name):
            self.history[name].pop(0)
            self.history_coords[name].pop(0)
            self.__square_diff__[name].pop(0)
            self.imgs[name].pop(0)
        self.history[name].append(square)
        self.history_coords[name].append(position)
        self.imgs[name].append(img)
        self.__square_diff__[name].append(
            list(list1 - list2 for list1, list2 in zip(self.square_ref[name], square)))

    def queueIsFull(self, name):
        return len(self.history[name]) == self.queue_size

    def isCriticalDamaged(self, name):
        if self.queueIsFull(name):
            return self.__calculateCriticity__(name)
        else:
            return (None,None)

    def __calculateCriticity__(self, name):
        loss = self.critical_loss
        square_ref = self.square_ref[name]
        square_diff = self.__square_diff__[name]
        coords = self.history_coords[name]
        imgs = self.imgs[name]
        percentage = []
        boolean = []

        # calculate % of loose
        for comb in zip([square_ref]*len(square_diff), square_diff):
            p, b = [], []
            for i in range(len(comb[0])):
                p.append((comb[1][i]*100)/comb[0][i])
                b.append(p[-1] > loss)
            percentage.append(p)
            boolean.append(b)

        perc_by_tooth = list(map(list, zip(*percentage)))
        bool_by_tooth = list(map(list, zip(*boolean)))
        coords_by_tooth = list(map(list, zip(*coords)))
        res = []
        img_idx = []

        # determine the accurate teeth area by selecting the most repeatable queue index
        for i, tooth in enumerate(bool_by_tooth):
            t = tooth.count(True)
            f = tooth.count(False)
            if f >= t:
                m_i = [i for i,v in enumerate(perc_by_tooth[i]) if v < loss]
            else:
                m_i = [i for i,v in enumerate(perc_by_tooth[i]) if v >= loss]
            img_idx.extend(m_i)
        
        lst_idx = Counter(img_idx)
        idx = max(sorted(img_idx,reverse = False), key=lst_idx.get)
        img = imgs[idx]

        for i, tooth in enumerate(bool_by_tooth):
            v = perc_by_tooth[i][idx]
            res.append((v >= loss, perc_by_tooth[i][idx], coords_by_tooth[i][idx]))

        return (res,img)
