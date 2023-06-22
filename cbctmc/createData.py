import os

import numpy as np


def run():
    arr = [
        "022",
        "024",
        "032",
        "033",
        "068",
        "069",
        "074",
        "078",
        "091",
        "092",
        "104",
        "106",
        "109",
        "115",
        "116",
        "121",
        "124",
        "132",
        "142",
        "145",
        "146",
    ]
    for id in arr:
        for i in range(10):
            for j in range(2):
                j = j * 5
                seed = np.random.randint(10, 123455463)
                os.system(
                    "run_mc --path_ct_in /home/crohling/data/ct/4d_ct_lung_uke_artifact_free/{}".format(
                        id
                    )
                    + "_4DCT_Lunge_amplitudebased_complete --filename_ct_in phase_0{}.nii ".format(
                        j
                    )
                    + "--path_out /home/crohling/data/results/low_4.8e8/low_pat{}".format(
                        id
                    )
                    + "_phase0"
                    + str(j)
                    + f"_run_{i:02d}"
                    + " --filename low_pat{}".format(id)
                    + "_phase0"
                    + str(j)
                    + f"_run_{i:02d}"
                    + " --no_sim 90 --photons 4.8e8 --gpu_id 0 "
                    "--random_seed {} --speed_up True --combine_photons False".format(
                        seed
                    )
                )


if __name__ == "__main__":
    run()
