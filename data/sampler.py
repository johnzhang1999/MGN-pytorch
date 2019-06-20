import random
import collections
from torch.utils.data import sampler

class RandomSampler(sampler.Sampler):
    def __init__(self, data_source, batch_id, batch_image):
        super(RandomSampler, self).__init__(data_source)

        self.data_source = data_source
        self.batch_image = batch_image
        self.batch_id = batch_id
        self.unique_ids = []
        self._id2index = collections.defaultdict(list)
        for idx, path in enumerate(data_source.imgs):
            _id = data_source.id(path)
            self._id2index[_id].append(idx)
            self.unique_ids.append(_id)
        # n = set()
        # for key,val in self._id2index.items():
        #     if len(val) == 0:
        #         n.add(key)
        # if len(n) == 0: print('NONE ARE EMPTY')

    def __iter__(self):
        # unique_ids = self.data_source.unique_ids
        random.shuffle(self.unique_ids)

        imgs = []
        for _id in self.unique_ids:
            imgs.extend(self._sample(self._id2index[_id], self.batch_image))
        return iter(imgs)

    def __len__(self):
        return len(self._id2index) * self.batch_image

    @staticmethod
    def _sample(population, k):
        if len(population) < k:
            population = population * k
        # print('population: {}, k: {}'.format(population,k))
        return random.sample(population, k)