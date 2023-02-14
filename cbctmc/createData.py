import os
import numpy as np


def run():
    arr = ["022", "024", "032", "033", "068", "069", "074", "078", "091", "092",
           "104", "106", "109", "115", "116", "121", "124", "132", "142", "145", "146"]
    for id in arr:
            for j in range(2):
                j = j*5
                seed = np.random.randint(10, 123455463)
                os.system("run_mc --path_ct_in /home/crohling/data/ct/4d_ct_lung_uke_artifact_free/{}".format(id) +
                          "_4DCT_Lunge_amplitudebased_complete --filename_ct_in phase_0{}.nii ".format(j) +
                          "--path_out /home/crohling/data/results/HIGH_2/HIGH_pat{}".format(id) + "_phase0" + str(j) +
                          " --filename low_pat{}".format(id) + "_phase0" + str(j) +
                          " --no_sim 90 --photons 2.4e9 --gpu_id 1 "
                          "--random_seed {} --speed_up True --combine_photons False".format(seed))


if __name__ == '__main__':
    run()