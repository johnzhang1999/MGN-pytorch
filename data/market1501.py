from data.common import list_pictures
import os.path as osp
import csv

from torch.utils.data import dataset
from torchvision.datasets.folder import default_loader

class Market1501(dataset.Dataset):
    def __init__(self, args, transform, dtype):

        self.transform = transform
        self.loader = default_loader

        data_path = args.datadir
        if dtype == 'train':
            data_path = osp.join(data_path, 'Info/sa_train.csv')
            self.lib = process_dir(data_path,relabel=True)
        elif dtype == 'test':
            data_path = osp.join(data_path, 'Info/sa_gallery.csv')
            self.lib = process_dir(data_path,relabel=False)
        else:
            data_path = osp.join(data_path, 'Info/sa_query.csv')
            self.lib = process_dir(data_path,relabel=False)
        
        self.data_root = args.datadir
        
        

        # self.imgs = [path for path in list_pictures(data_path) if self.id(path) != -1]

        # self._id2label = {_id: idx for idx, _id in enumerate(self.unique_ids)}

    def process_dir(self, dir_path, relabel=False):
        ret = []
        real_ret = []
        pid_container = set()
        with open(osp.join(dir_path)) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                filename,pid,cam = row[0],row[1],row[2]
                pid_container.add(pid)
                ret.append((osp.join(self.data_root,'Image',filename),pid,int(cam)))
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        # print(pid2label)
        if relabel:
            for path,pid,cam in ret:
                real_ret.append((path,pid2label[pid],cam))
        else:
            for path,pid,cam in ret:
                real_ret.append((path,int(pid),cam))
        return real_ret

    def __getitem__(self, index):

        path,pid,_ = self.lib[index]

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, pid

    def __len__(self):
        return len(self.lib)

    @staticmethod
    def id(file_path):
        """
        :param file_path: unix style file path
        :return: person id
        """
        return int(file_path.split('/')[-1].split('_')[0])

    @staticmethod
    def camera(file_path):
        """
        :param file_path: unix style file path
        :return: camera id
        """
        return int(file_path.split('/')[-1].split('_')[2][1])

    @property
    def ids(self):
        """
        :return: person id list corresponding to dataset image paths
        """
        return [img[1] for img in self.lib]

    @property
    def unique_ids(self):
        """
        :return: unique person ids in ascending order
        """
        return [img[1] for img in self.lib]

    @property
    def cameras(self):
        """
        :return: camera id list corresponding to dataset image paths
        """
        return [img[2] for img in self.lib]
