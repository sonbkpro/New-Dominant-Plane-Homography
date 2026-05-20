from torch.utils.data import Dataset
import  numpy as np
import cv2, torch
import os
import random


def make_mesh(patch_w,patch_h):
    x_flat = np.arange(0,patch_w)
    x_flat = x_flat[np.newaxis,:]
    y_one = np.ones(patch_h)
    y_one = y_one[:,np.newaxis]
    x_mesh = np.matmul(y_one , x_flat)

    y_flat = np.arange(0,patch_h)
    y_flat = y_flat[:,np.newaxis]
    x_one = np.ones(patch_w)
    x_one = x_one[np.newaxis,:]
    y_mesh = np.matmul(y_flat,x_one)
    return x_mesh,y_mesh


class TrainDataset(Dataset):
    def __init__(self, data_path, exp_path, patch_w=560, patch_h=315, rho=16,
                 return_invalid=True, return_triplet=False):

        self.imgs = open(data_path, 'r').readlines()
        self.records = [line.strip().split(' ') for line in self.imgs if line.strip()]
        self.mean_I = np.reshape(np.array([118.93, 113.97, 102.60]), (1, 1, 3))
        self.std_I = np.reshape(np.array([69.85, 68.81, 72.45]), (1, 1, 3))

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.WIDTH = 640
        self.HEIGHT = 360
        self.rho = rho
        self.x_mesh, self.y_mesh = make_mesh(self.patch_w, self.patch_h)
        self.train_path = os.path.join(exp_path, 'Data/Train/')
        self.return_invalid = return_invalid
        self.return_triplet = return_triplet
        self.video_to_paths = self._build_video_index()

    def _build_video_index(self):
        video_to_paths = {}
        for pair in self.records:
            for rel_path in pair[:2]:
                video = rel_path.split('/')[0]
                video_to_paths.setdefault(video, set()).add(rel_path)
        indexed = {}
        for video, paths in video_to_paths.items():
            indexed[video] = sorted(paths, key=self._frame_number)
        return indexed

    @staticmethod
    def _frame_number(rel_path):
        name = os.path.splitext(os.path.basename(rel_path))[0]
        try:
            return int(name.split('_')[-1])
        except ValueError:
            return 0

    def _load_gray_norm(self, rel_path):
        img = cv2.imread(os.path.join(self.train_path, rel_path))
        if img is None:
            raise FileNotFoundError(os.path.join(self.train_path, rel_path))
        height, width = img.shape[:2]
        if height != self.HEIGHT or width != self.WIDTH:
            img = cv2.resize(img, (self.WIDTH, self.HEIGHT))
        img = (img - self.mean_I) / self.std_I
        img = np.mean(img, axis=2, keepdims=True)
        return np.transpose(img, [2, 0, 1])

    def _sample_crop(self):
        x = np.random.randint(self.rho, self.WIDTH - self.rho - self.patch_w)
        y = np.random.randint(self.rho, self.HEIGHT - self.rho - self.patch_h)
        return x, y

    def _make_pair(self, img_1, img_2, x=None, y=None):
        if x is None or y is None:
            x, y = self._sample_crop()
        org_img = np.concatenate([img_1, img_2], axis=0)
        input_tesnor = org_img[:, y: y + self.patch_h, x: x + self.patch_w]

        y_t_flat = np.reshape(self.y_mesh, (-1))
        x_t_flat = np.reshape(self.x_mesh, (-1))
        patch_indices = (y_t_flat + y) * self.WIDTH + (x_t_flat + x)

        h4p = np.reshape([
            (x, y),
            (x, y + self.patch_h),
            (self.patch_w + x, self.patch_h + y),
            (x + self.patch_w, y),
        ], (-1))

        return {
            'org': torch.tensor(org_img),
            'input': torch.tensor(input_tesnor),
            'patch_indices': torch.tensor(patch_indices),
            'h4p': torch.tensor(h4p),
        }

    def _sample_invalid_second(self, index):
        src_video = self.records[index][0].split('/')[0]
        candidates = [i for i, pair in enumerate(self.records)
                      if pair[1].split('/')[0] != src_video]
        if not candidates:
            candidates = [i for i in range(len(self.records)) if i != index]
        return self.records[random.choice(candidates)][1]

    def _sample_triplet_paths(self, index):
        video = self.records[index][0].split('/')[0]
        paths = self.video_to_paths.get(video, [])
        if len(paths) < 3:
            return None
        pos = random.randint(0, len(paths) - 3)
        return paths[pos], paths[pos + 1], paths[pos + 2]

    def __getitem__(self, index):
        img_names = self.records[index]
        img_1 = self._load_gray_norm(img_names[0])
        img_2 = self._load_gray_norm(img_names[1])
        valid = self._make_pair(img_1, img_2)

        sample = {
            'org': valid['org'],
            'input': valid['input'],
            'patch_indices': valid['patch_indices'],
            'h4p': valid['h4p'],
        }

        if self.return_invalid:
            invalid_2 = self._load_gray_norm(self._sample_invalid_second(index))
            invalid = self._make_pair(img_1, invalid_2)
            sample.update({
                'invalid_org': invalid['org'],
                'invalid_input': invalid['input'],
                'invalid_patch_indices': invalid['patch_indices'],
                'invalid_h4p': invalid['h4p'],
            })

        if self.return_triplet:
            triplet_paths = self._sample_triplet_paths(index)
            if triplet_paths is not None:
                img_t = self._load_gray_norm(triplet_paths[0])
                img_t1 = self._load_gray_norm(triplet_paths[1])
                img_t2 = self._load_gray_norm(triplet_paths[2])
                triplet_available = 1.0
            else:
                img_t = img_1
                img_t1 = img_2
                img_t2 = img_2
                triplet_available = 0.0
            x, y = self._sample_crop()
            pair_01 = self._make_pair(img_t, img_t1, x, y)
            pair_12 = self._make_pair(img_t1, img_t2, x, y)
            pair_02 = self._make_pair(img_t, img_t2, x, y)
            sample.update({
                'triplet_available': torch.tensor(triplet_available),
                'triplet01_org': pair_01['org'],
                'triplet01_input': pair_01['input'],
                'triplet01_patch_indices': pair_01['patch_indices'],
                'triplet01_h4p': pair_01['h4p'],
                'triplet12_org': pair_12['org'],
                'triplet12_input': pair_12['input'],
                'triplet12_patch_indices': pair_12['patch_indices'],
                'triplet12_h4p': pair_12['h4p'],
                'triplet02_org': pair_02['org'],
                'triplet02_input': pair_02['input'],
                'triplet02_patch_indices': pair_02['patch_indices'],
                'triplet02_h4p': pair_02['h4p'],
            })

        return sample

    def __len__(self):
        return len(self.records)


class TestDataset(Dataset):
    def __init__(self, data_path, patch_w=560, patch_h=315, rho=16,
                 WIDTH=640, HEIGHT=360, coord_dir=None):
        self.mean_I = np.reshape(np.array([118.93, 113.97, 102.60]), (1, 1, 3))
        self.std_I = np.reshape(np.array([69.85, 68.81, 72.45]), (1, 1, 3))

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.rho = rho
        self.x_mesh, self.y_mesh = make_mesh(self.patch_w,self.patch_h)

        self.work_dir = os.path.join(data_path, 'Data')
        self.pair_list = list(open(os.path.join(self.work_dir, 'Test_List.txt')))
        print(len(self.pair_list))
        self.img_path = os.path.join(self.work_dir, 'Test/')
        if coord_dir is None:
            coord_v2 = os.path.join(self.work_dir, 'Coordinate-v2/Coordinate-v2/')
            coord_v1 = os.path.join(self.work_dir, 'Coordinate/')
            coord_dir = coord_v2 if os.path.isdir(coord_v2) else coord_v1
        self.npy_path = coord_dir if coord_dir.endswith(os.sep) else coord_dir + os.sep

    def __getitem__(self, index):

        img_pair = self.pair_list[index]
        pari_id = img_pair.split(' ')
        npy_id = pari_id[0].split('/')[1] + '_' + pari_id[1].split('/')[1][:-1] + '.npy'
        npy_id = self.npy_path + npy_id
        video_name = img_pair.split('/')[0]

        # load img1
        if pari_id[0][-1] == 'M':
            img_1 = cv2.imread(self.img_path + pari_id[0][:-2])
        else:
            img_1 = cv2.imread(self.img_path + pari_id[0])

        # load img2
        if pari_id[1][-2] == 'M':
            img_2 = cv2.imread(self.img_path + pari_id[1][:-3])
        else:
            img_2 = cv2.imread(self.img_path + pari_id[1][:-1])
        
        height, width = img_1.shape[:2]
 
        if height != self.HEIGHT or width != self.WIDTH:
            img_1 = cv2.resize(img_1, (self.WIDTH, self.HEIGHT))

        print_img_1 = img_1.copy()
        print_img_1 = np.transpose(print_img_1, [2, 0, 1])

        img_1 = (img_1 - self.mean_I) / self.std_I
        img_1 = np.mean(img_1, axis=2, keepdims=True)
        img_1 = np.transpose(img_1, [2, 0, 1])

        height, width = img_2.shape[:2]

        if height != self.HEIGHT or width != self.WIDTH:
            img_2 = cv2.resize(img_2, (self.WIDTH, self.HEIGHT))

        print_img_2 = img_2.copy()
        print_img_2 = np.transpose(print_img_2, [2, 0, 1])
        img_2 = (img_2 - self.mean_I) / self.std_I
        img_2 = np.mean(img_2, axis=2, keepdims=True)
        img_2 = np.transpose(img_2, [2, 0, 1])
        org_img = np.concatenate([img_1, img_2], axis=0)
        WIDTH = org_img.shape[2]
        HEIGHT = org_img.shape[1]

        x = np.random.randint(self.rho, WIDTH - self.rho - self.patch_w)
        x = 40  # patch should in the middle of full img when testing
        y = np.random.randint(self.rho, HEIGHT - self.rho - self.patch_h)
        y = 23  # patch should in the middle of full img when testing
        input_tesnor = org_img[:, y: y + self.patch_h, x: x + self.patch_w]

        y_t_flat = np.reshape(self.y_mesh, [-1])
        x_t_flat = np.reshape(self.x_mesh, [-1])
        patch_indices = (y_t_flat + y) * WIDTH + (x_t_flat + x)

        top_left_point = (x, y)
        bottom_left_point = (x, y + self.patch_h)
        bottom_right_point = (self.patch_w + x, self.patch_h + y)
        top_right_point = (x + self.patch_w, y)
        four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]

        four_points = np.reshape(four_points, (-1))

        return (org_img, input_tesnor, patch_indices, four_points,print_img_1, print_img_2, video_name, npy_id)

    def __len__(self):

        return len(self.pair_list)
