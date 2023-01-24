import os

import numpy as np


def run2():
    for i in range(300):
        seed = np.random.randint(10, 123455463)
        os.system("run_mc --path_ct_in /home/crohling/data/ct/4d_ct_lung_uke_artifact_free/022" +
                  "_4DCT_Lunge_amplitudebased_complete --filename_ct_in phase_00.nii " +
                  f"--path_out /home/crohling/data/results/test/test_{i:02d}" +
                  f" --filename test_{i:02d}" + " --no_sim 1 --photons 5e7 --gpu_id 1 "
                  "--random_seed {} --speed_up True --combine_photons False".format(seed))

def run():
    # "022", "024", "032", "033", "068", "069", "074", "078", "091", "092",
    arr = ["104", "106", "109", "115", "116", "121", "124", "132", "142", "145", "146"]
    for id in arr:
        for i in range(15):
            for j in range(2):
                j = j*5
                seed = np.random.randint(10, 123455463)
                os.system("run_mc --path_ct_in /home/crohling/data/ct/4d_ct_lung_uke_artifact_free/{}".format(id) +
                          "_4DCT_Lunge_amplitudebased_complete --filename_ct_in phase_0{}.nii ".format(j) +
                          "--path_out /home/crohling/data/results/low_pat{}".format(id) + "_phase0" + str(j) +
                          f"_run_{i:02d}" +
                          " --filename low_pat{}".format(id) + "_phase0" + str(j) +
                          f"_run_{i:02d}" + " --no_sim 90 --photons 5e7 --gpu_id 0 "
                          "--random_seed {} --speed_up True --combine_photons False".format(seed))
        for j in (0, 5):

            seed = np.random.randint(10, 123455463)
            os.system("run_mc --path_ct_in /home/crohling/data/ct/4d_ct_lung_uke_artifact_free/{}".format(id) +
                      "_4DCT_Lunge_amplitudebased_complete --filename_ct_in phase_0{}.nii ".format(j) +
                      "--path_out /home/crohling/data/results/HIGH_pat{}".format(id) + "_phase0" + str(j) +
                      " --filename HIGH_pat{}".format(id) + "_phase0" + str(j) +
                      " --no_sim 90 --gpu_id 0 "
                      "--random_seed {} --speed_up True".format(seed))


if __name__ == '__main__':
    run2()