import os

import numpy as np


def run():
    for i in range(10):
        for j in range(2):
            j = j*5
            seed = np.random.randint(10, 123455463)
            os.system("run_mc --path_ct_in /home/crohling/data/ct --filename_ct_in bin_0{}.nii".format(j) +
                      "--path_out /home/crohling/data/results/low_{}".format("bin_0" + str(j) + "_" + str(i)) +
                      " --filename mc --no_sim 90 --photons 5e7 --gpu_id 3 "
                      "--random_seed {} --normalize False --combine_photons False".format(seed))
    for j in range(2):
        j = j * 5
        seed = np.random.randint(10, 123455463)
        os.system("run_mc --path_ct_in /home/crohling/data/ct --filename_ct_in bin_0{}.nii".format(j) +
                  "--path_out /home/crohling/data/results/high_{} --filename mc ".format("bin_0" + str(j)) +
                  "--no_sim 90 --photons 2.4e9 --gpu_id 3 "
                  "--random_seed {} --normalize False".format(seed))


if __name__ == '__main__':
    run()