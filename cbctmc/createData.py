import os

import numpy as np


def run():
    arr = ["022", "024", "032", "033", "068", "069", "074", "078", "091", "092", "104", "106", "109", "115", "116",
           "121", "124", "132", "142", "145", "146"]
    for id in arr:
        for i in range(30):
            for j in range(2):
                j = j*5
                seed = np.random.randint(10, 123455463)
                os.system("run_mc --path_ct_in /home/crohling/data/ct --filename_ct_in bin_0{}.nii ".format(j) +
                          "--path_out /home/crohling/data/results/low_{}".format("bin_0" + str(j) + "_" + str(i)) +
                          " --filename mc --no_sim 90 --photons 5e7 --gpu_id 3 "
                          "--random_seed {} --normalize False --combine_photons False".format(seed))
    for j in range(2):
        j = j * 5
        seed = np.random.randint(10, 123455463)
        os.system("run_mc --path_ct_in /home/crohling/data/ct --filename_ct_in bin_0{}.nii ".format(j) +
                  "--path_out /home/crohling/data/results/high_{} --filename mc ".format("bin_0" + str(j)) +
                  "--photons 2.4e9 --gpu_id 3 --no_sim 90 "
                  "--random_seed {} --normalize False".format(seed))


if __name__ == '__main__':
    run()

    "run_mc --path_ct_in /home/crohling/data/ct/4d_ct_lung_uke_artifact_free/022_4DCT_Lunge_amplitudebased_complete --filename_ct_in phase_00.nii --path_out /home/crohling/data/results/low_022_test --filename patient_22 --photons 5e7 --gpu_id 3  --random_seed 42 --speed_up True --combine_photons False"