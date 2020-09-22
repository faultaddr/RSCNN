import torch
import torch.utils.data as data
import numpy as np
import os, sys, h5py

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

def _get_data_files(list_filename):
    with open(list_filename) as f:
        return [line.rstrip()[0:] for line in f]

def _load_data_file(name):
    f = h5py.File(name)
    data = f['data'][:]
    label = f['label'][:]
    return data, label

def pc_normalize(pc):
    # pc.shape[0] is the number of points
    # pc.shape[1] is the number of feature dimensions
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    dist = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / dist
    return pc

class ModelNet40Cls(data.Dataset):

    def __init__(
            self, num_points, root, transforms=None, train=True
    ):
        super().__init__()

        self.transforms = transforms

        root = os.path.abspath(root)
        self.folder = "modelnet40_ply_hdf5_2048"
        self.data_dir = root
        print(self.data_dir)
        self.train, self.num_points = train, num_points
        if self.train:
            self.files =  _get_data_files( \
                os.path.join(self.data_dir, 'train_files.txt'))
        else:
            self.files =  _get_data_files( \
                os.path.join(self.data_dir, 'test_files.txt'))

        point_list, label_list = [], []
        for f in self.files:
            print(f)
            points, labels = _load_data_file(os.path.join(self.data_dir,f))
            points = pc_normalize(points)
            point_list.append(points)
            label_list.append(labels)

        self.points = np.concatenate(point_list, 0)
        self.labels = np.concatenate(label_list, 0)
        print(np.max(self.labels))

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.points.shape[1])   # 2048
        if self.train:
            np.random.shuffle(pt_idxs)

        current_points = self.points[idx, pt_idxs].copy()
        label = torch.from_numpy(self.labels[idx]).type(torch.LongTensor)

        if self.transforms is not None:
            current_points = self.transforms(current_points)

        return current_points, label

    def __len__(self):
        return self.points.shape[0]

if __name__ == "__main__":
    from torchvision import transforms
    import data_utils as d_utils

    transforms = transforms.Compose([
        d_utils.PointcloudToTensor(),
        d_utils.PointcloudRotate(axis=np.array([1,0,0])),
        d_utils.PointcloudScale(),
        d_utils.PointcloudTranslate(),
        d_utils.PointcloudJitter()
    ])
    dset = ModelNet40Cls(16, "./", train=True, transforms=transforms)
    print(dset[0][0])
    print(dset[0][1])
    print(len(dset))
    dloader = torch.utils.data.DataLoader(dset, batch_size=32, shuffle=True)
