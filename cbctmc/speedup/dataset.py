import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import scipy


class CustomImageDataset(Dataset):
    def __init__(self, low_path, high_path):
        self.low_path = low_path
        self.high_path = high_path

    def __len__(self):
        return len(self.low_path)

    def __getitem__(self, idx):
        low = np.load(self.low_path[idx])
        high = np.load(self.high_path[idx])

        return {
            'low_photon': low,
            'high_photon': high,
            'low_photon_filepath': self.low_path[idx],
            'high_photon_filepath': self.high_path[idx]
        }


def createDataset():

    arr = ["022", "024", "032", "033", "068", "069", "074", "078", "091"]
    # , "092", "104", "106", "109", "115", "116","121", "124", "132", "142", "145", "146"]
    low = []
    high = []
    for id in arr:
        for i in range(15):
            for j in (0, 5):
                for k in range(90):
                    low.append("/home/crohling/amalthea/data/results/low_pat{}".format(id) + "_phase0" + str(j) +
                          f"_run_{i:02d}/low_pat{id}" + "_phase0" + str(j) +
                          f"_run_{i:02d}" + f"_proj_{k:02d}.npy")

                    high.append("/home/crohling/amalthea/data/results/HIGH_pat{}".format(id) + "_phase0" + str(j) +
                     "/HIGH_pat{}".format(id) + "_phase0" + str(j) + f"_proj_{k:02d}.npy")
    return CustomImageDataset(low, high)


def createTestDataset(i):
    low = f"/home/crohling/amalthea/data/results/test/test_{i:02d}" + f"/test_{i:02d}_proj_00.npy"
    return low


if __name__ == '__main__':
    dataset = createDataset()
    print(len(dataset))
    data = []
    for i in range(300):
        data.append(np.load(createTestDataset(i))[..., 0])

    en = np.loadtxt("/home/crohling/Documents/MC-GPU_v1.3_RELEASE_2012-12-12/neuesSpectra.spc")

    norm = scipy.integrate.simps(en[:, 1], en[:, 0])
    mean_energy = scipy.integrate.simps(en[:, 1]/norm*en[:, 0], en[:, 0])
    data = np.array(data)
    data = data * (0.006024 * 924 * 384) * 5e7/mean_energy
    data = np.array(data)
    mean = np.mean(data, axis=0)
    var = np.var(data, axis=0)

    plt.plot(mean.flatten(), var.flatten(), "bo")




    plt.plot(np.arange(0, 30, 0.01), 1/5*np.arange(0, 30, 0.01), "red")
    plt.show()

    print(mean.shape)

    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

