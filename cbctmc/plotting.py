import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from scipy.optimize import curve_fit


def analyzeSpec():
    sp = np.loadtxt("SpectraTit125")
    plt.plot(sp[:, 0] * 1e3, sp[:, 1], "bo")
    en = np.arange(8, 125.5, 0.5) * 1e3
    new = np.interp(en, sp[:, 0] * 1e3, sp[:, 1])
    plt.plot(en, new, "r+")
    plt.show()
    print(en, new)
    np.savetxt("neuesSpectra", np.stack((en, new), axis=1))


def getLabel(i):
    if i == 1:
        return "Air"
    if i == 2:
        return "Teflon"
    if i == 3:
        return "Delrin"
    if i == 4:
        return "Bone 20%"
    if i == 5:
        return "Acryl"
    if i == 6:
        return "PMP"
    if i == 7:
        return "Bone 50%"
    if i == 8:
        return "LDPE"
    if i == 9:
        return "Polystrene"
    if i == 10:
        return "Water"
    if i == 11:
        return "Air-outside"
    return "Water"


def analyzeCat():
    cbct_real = sitk.ReadImage(
        "/home/crohling/Documents/ct-data/2022-12-01_142914/normalized_recon.mha"
    )
    cbct_real_seg = sitk.ReadImage(
        "/home/crohling/Documents/ct-data/2022-12-01_142914/catphan_normalized_seg.nii.gz"
    )
    cbct_sim = sitk.ReadImage(
        "/home/crohling/Documents/MC-GPU_v1.3_RELEASE_2012-12-12/Simulation/catphan_06/recon5.mha"
    )
    cbct_sim_seg = sitk.ReadImage(
        "/home/crohling/Documents/MC-GPU_v1.3_RELEASE_2012-12-12/Simulation/catphan_06/catphan_sim_seg.nii.gz"
    )
    cbct_real = sitk.GetArrayFromImage(cbct_real)
    cbct_real_seg = sitk.GetArrayFromImage(cbct_real_seg)
    cbct_sim = sitk.GetArrayFromImage(cbct_sim)
    cbct_sim_seg = sitk.GetArrayFromImage(cbct_sim_seg)
    print(cbct_real.shape, cbct_real.dtype)
    real = []
    sim = []
    x = np.ones((256, 256, 256)) * -2000
    waterreal = []
    watersim = []
    for i in range(19):
        i = i + 1
        if i < 12:
            data = np.where(cbct_real_seg == i, cbct_real, x)
            data = data[(data != -2000)]
            real.append(data.mean())
            print(i, data.mean())
        else:
            real.append(real[9])
            data = np.where(cbct_real_seg == 10, cbct_real, x)
            data = data[(data != -2000)]
            print(i, data.mean())
        data_sim = np.where(cbct_sim_seg == i, cbct_sim, x)
        data_sim = data_sim[(data_sim != -2000)]
        data_sim = data_sim[: len(data)]
        # bins = np.linspace(-1, 1, 100)
        plt.hist(data_sim, 30, alpha=0.5, label="Simulation")
        plt.hist(data, 30, alpha=0.5, label="Real")
        plt.legend(loc="upper right")
        plt.title(getLabel(i))
        plt.show()

        sim.append(data_sim.mean())
        if i == 10:
            waterreal = data
            watersim = data_sim
    sim = np.array(sim)
    real = np.array(real)
    plt.plot(real, sim, "bo", label="Mean Real")
    # plt.plot(np.arange(1,12,1), sim, "r+", label="Simultion Mean")
    text = ""
    for i in range(11):
        i += 1
        text += "x=" + str(i) + " => " + str(getLabel(i)) + "\n"
    plt.legend(loc="upper right")
    plt.show()
    plt.plot(np.arange(1, 20, 1), sim, "r+", label="Sim mean")
    plt.plot(np.arange(1, 20, 1), real, "bo", label="Real mean")
    plt.annotate(text, xy=(0.10, 0.05), xycoords="axes fraction")
    plt.legend(loc="upper right")
    plt.show()
    plt.plot(
        np.arange(2, 20, 1),
        ((sim - real) / (real))[1:],
        "r+",
        label="relative difference",
    )
    plt.legend(loc="upper right")
    plt.show()


def linear(x, a, b):
    return a * x + b


def zylinder(v, m, r, h):
    if m[2] - h / 2 <= v[2] <= m[2] + h / 2:
        if (v[0] - m[0]) ** 2 + (v[1] - m[1]) ** 2 <= r**2:
            return True
    return False


def voxPhantom():
    vox = np.empty((512, 512, 414), dtype="U21")
    for x in range(512):
        print(x)
        for y in range(512):
            for z in range(414):
                v = [x, y, z]
                if zylinder(v, [256, 306, 258.5], 6, 53):
                    vox[x, y, z] = "1 0.0012"
                elif zylinder(v, [256, 206, 258.5], 6, 53):
                    vox[x, y, z] = "1 0.0012"
                elif zylinder(v, [206, 256, 258.5], 6, 53):
                    vox[x, y, z] = "3 0.92"
                elif zylinder(v, [306, 256, 258.5], 6, 53):
                    vox[x, y, z] = "7 1.42"
                elif zylinder(v, [281, 212.7, 258.5], 6, 53):
                    vox[x, y, z] = "9 2.16"
                elif zylinder(v, [299.3, 281, 258.5], 6, 53):
                    vox[x, y, z] = "6 1.14"
                elif zylinder(v, [281, 299.3, 258.5], 6, 53):
                    vox[x, y, z] = "5 1.18"
                elif zylinder(v, [231, 299.3, 258.5], 6, 53):
                    vox[x, y, z] = "4 1.03"
                elif zylinder(v, [212.7, 231, 258.5], 6, 53):
                    vox[x, y, z] = "8 1.4"
                elif zylinder(v, [231, 212.7, 258.5], 6, 53):
                    vox[x, y, z] = "2 0.83"
                elif zylinder(v, [256, 256, 177 + 38], 86, 354):
                    vox[x, y, z] = "10 1.0"
                else:
                    vox[x, y, z] = "1 0.0012"
    print(str(vox))
    np.save("vox_phantom", vox)


def readDoseImage(filepath, det_pixel_y, det_pixel_x):
    # huge time waste next line, maybe mulitprocessing
    nonsc = np.loadtxt(filepath, dtype="float")
    nonsc = nonsc[:, 0] + nonsc[:, 1] + nonsc[:, 2] + nonsc[:, 3]
    nonsc = np.reshape(nonsc, (int(nonsc.size / det_pixel_y), -1))
    nonsc = nonsc[:, 0:det_pixel_x]
    nonsc = np.flip(nonsc, 0)
    return nonsc


def nonNormalized(sim_path, sim_filename):
    proj = []
    det_pixel_x = 512
    lat_displacement = -160
    spacing = 0.776
    det_pix_size = 0.776
    det_pixel_x_halffan = int(det_pixel_x + np.abs(lat_displacement) / det_pix_size * 2)
    for i in range(894):
        dat = "000" + str(i)
        print(str(dat[-4:]))
        dat = "_" + str(dat[-4:])
        proj.append(
            readDoseImage(
                sim_path + "/" + sim_filename + dat, det_pixel_x_halffan, det_pixel_x
            )
        )
    proj = np.array(proj)
    proj_im = sitk.GetImageFromArray(proj)
    proj_im.SetSpacing((spacing, spacing, 1))
    proj_im.SetOrigin(
        (
            int(-proj_im.GetSize()[0] * proj_im.GetSpacing()[0] / 2),
            int(-proj_im.GetSize()[1] * proj_im.GetSpacing()[1] / 2),
            1,
        )
    )
    sitk.WriteImage(proj_im, "nonNormalized.mha")


def oneDsqrt(x, a):
    return 1 / np.sqrt(x) * a


def getNPhoton(para):
    cbct_real = sitk.ReadImage(
        "/home/crohling/Documents/ct-data/2022-12-01_142914/normalized_recon.mha"
    )
    cbct_real_seg = sitk.ReadImage(
        "/home/crohling/Documents/ct-data/2022-12-01_142914/catphan_normalized_seg.nii.gz"
    )
    cbct_sim_87e8 = sitk.ReadImage(
        "/home/crohling/Documents/MC-GPU_v1.3_RELEASE_2012-12-12/Simulation/catphan_03/recon2.mha"
    )
    cbct_sim_1e8 = sitk.ReadImage(
        "/home/crohling/Documents/MC-GPU_v1.3_RELEASE_2012-12-12/Simulation/catphan_04/recon.mha"
    )
    cbct_sim_5e8 = sitk.ReadImage(
        "/home/crohling/Documents/MC-GPU_v1.3_RELEASE_2012-12-12/Simulation/catphan_05/recon.mha"
    )
    cbct_sim_24e8 = sitk.ReadImage(
        "/home/crohling/Documents/MC-GPU_v1.3_RELEASE_2012-12-12/Simulation/catphan_06/recon.mha"
    )
    cbct_sim_seg = sitk.ReadImage(
        "/home/crohling/Documents/MC-GPU_v1.3_RELEASE_2012-12-12/Simulation/catphan_03/catphan_sim_seg.nii.gz"
    )
    cbct_real = sitk.GetArrayFromImage(cbct_real)
    cbct_real_seg = sitk.GetArrayFromImage(cbct_real_seg)
    cbct_sim_1e8 = sitk.GetArrayFromImage(cbct_sim_1e8)
    cbct_sim_5e8 = sitk.GetArrayFromImage(cbct_sim_5e8)
    cbct_sim_87e8 = sitk.GetArrayFromImage(cbct_sim_87e8)
    cbct_sim_seg = sitk.GetArrayFromImage(cbct_sim_seg)
    cbct_sim_24e8_t = sitk.GetArrayFromImage(cbct_sim_24e8)
    new = sitk.GetImageFromArray(cbct_sim_24e8_t * para[0] + para[1])
    new.CopyInformation(cbct_sim_24e8)
    sitk.WriteImage(new, "../calibratedphantom.mha")
    cbct_sim_24e8 = cbct_sim_24e8_t

    x = np.ones((256, 256, 256)) * -2000
    data = np.where(cbct_real_seg == 10, cbct_real, x)
    data = data[(data != -2000)]
    data_sim_87e8 = np.where(cbct_sim_seg == 10, cbct_sim_87e8, x)
    data_sim_87e8 = data_sim_87e8[(data_sim_87e8 != -2000)]
    data_sim_1e8 = np.where(cbct_sim_seg == 10, cbct_sim_1e8, x)
    data_sim_1e8 = data_sim_1e8[(data_sim_1e8 != -2000)]
    data_sim_5e8 = np.where(cbct_sim_seg == 10, cbct_sim_5e8, x)
    data_sim_5e8 = data_sim_5e8[(data_sim_5e8 != -2000)]
    data_sim_24e8 = np.where(cbct_sim_seg == 10, cbct_sim_24e8, x)
    data_sim_24e8 = data_sim_24e8[(data_sim_24e8 != -2000)]
    data_sim_5e8 = data_sim_5e8[:15565]
    data_sim_1e8 = data_sim_1e8[:15565]
    data_sim_87e8 = data_sim_87e8[:15565]
    data_sim_24e8 = data_sim_24e8[:15565]
    plt.hist(
        data_sim_87e8 * para[0] + para[1], 30, alpha=0.5, label="Simulation N = 8.7e9"
    )
    plt.hist(
        data_sim_5e8 * para[0] + para[1], 30, alpha=0.5, label="Simulation N = 5e8"
    )
    plt.hist(
        data_sim_1e8 * para[0] + para[1], 30, alpha=0.5, label="Simulation N = 1e8"
    )
    plt.hist(
        data_sim_24e8 * para[0] + para[1], 30, alpha=0.5, label="Simulation N = 2.4e9"
    )
    plt.hist(data, 30, alpha=0.5, label="Real")
    plt.legend(loc="upper right")
    plt.show()
    n = np.array([1e8, 5e8, 2.4e9, 8.7e9])
    var = np.array(
        [
            np.std(data_sim_1e8 * para[0] + para[1]),
            np.std(data_sim_5e8 * para[0] + para[1]),
            np.std(data_sim_24e8 * para[0] + para[1]),
            np.std(data_sim_87e8 * para[0] + para[1]),
        ]
    )
    plt.plot(n, var, "bo")
    plt.show()

    par, cov = curve_fit(oneDsqrt, n, var)
    plt.plot(n, var, "bo")
    plt.plot(np.arange(7e7, 1e10, 1e2), oneDsqrt(np.arange(7e7, 1e10, 1e2), par))
    plt.show()
    plt.plot(n, var, "bo")
    plt.plot(np.arange(7e7, 1e10, 1e2), oneDsqrt(np.arange(7e7, 1e10, 1e2), par))
    plt.show()
    print("N = ", 1 / np.std(data) ** 2 * par**2)
    print("St Deviation real Scan: " + str(np.std(data)))
    print(
        "St Deviation simulated 2.4e9 Photons Scan: "
        + str(np.std(data_sim_24e8 * para[0] + para[1]))
    )
    print("Theoretical Std deviation 2.4e9 Photons" + str(1 / np.sqrt(2.4e9) * par))


def getSeg(i):
    i += 1
    if i == 1:
        return "Air"
    if i == 2:
        return "Lung"
    if i == 3:
        return "Spine"
    if i == 4:
        return "Heart"
    if i == 5:
        return "Breast"


def analyzePatient():
    cbct_real = sitk.ReadImage("/home/crohling/Downloads/2017-08-30_162441/recon.mha")
    print(cbct_real.GetSpacing())
    cbct_real_seg = sitk.ReadImage(
        "/home/crohling/Downloads/2017-08-30_162441/Segmentation/Untitled.nii.gz"
    )
    cbct_sim = sitk.ReadImage(
        "/home/crohling/Documents/MC-GPU_v1.3_RELEASE_2012-12-12/Simulation/bin_00_2/recon2.mha"
    )
    img_data_saver = cbct_sim
    cbct_sim_seg = sitk.ReadImage(
        "/home/crohling/Documents/MC-GPU_v1.3_RELEASE_2012-12-12/Simulation/bin_00_2/Untitled.nii.gz"
    )
    cbct_real = sitk.GetArrayFromImage(cbct_real)
    cbct_real_seg = sitk.GetArrayFromImage(cbct_real_seg)
    cbct_sim = sitk.GetArrayFromImage(cbct_sim)
    cbct_sim_seg = sitk.GetArrayFromImage(cbct_sim_seg)
    img = sitk.GetImageFromArray(cbct_sim)
    img.CopyInformation(img_data_saver)

    real = []
    sim = []
    x = np.ones((256, 256, 256)) * -2000
    waterreal = []
    watersim = []
    for i in range(5):
        i = i + 1
        data = np.where(cbct_real_seg == i, cbct_real, x)
        data = data[(data != -2000)]
        data_sim = np.where(cbct_sim_seg == i, cbct_sim, x)
        data_sim = data_sim[(data_sim != -2000)]
        n = data.shape[0]
        if data.shape[0] > data_sim.shape[0]:
            n = data_sim.shape[0]
            data = data[:n]
        else:
            data_sim = data_sim[:n]
        # bins = np.linspace(-1, 1, 100)
        if data.shape[0] > 1:
            plt.hist(data_sim, 30, alpha=0.5, label="Simulation")
            plt.hist(data, 30, alpha=0.5, label="Real")
            plt.legend(loc="upper right")
            plt.show()
            real.append(np.mean(data))
            sim.append(np.mean(data_sim))
    plt.plot(np.arange(5), sim, "bo", label="Sim Mean")
    plt.plot(np.arange(5), real, "r+", label="Real Mean")
    text = ""
    for i in range(5):
        text += "x=" + str(i) + " => " + str(getSeg(i)) + "\n"
    plt.annotate(text, xy=(0.10, 0.5), xycoords="axes fraction")
    plt.legend(loc="upper right")
    plt.show()


def flipRecon():
    cbct_sim = sitk.ReadImage(
        "/home/crohling/Documents/MC-GPU_v1.3_RELEASE_2012-12-12/Simulation/bin_00_2/recon.mha"
    )
    npimg = sitk.GetArrayFromImage(cbct_sim)
    npimg = np.rot90(npimg, 0)
    npimg = npimg.swapaxes(0, 2)
    npimg = sitk.GetImageFromArray(npimg)
    npimg.CopyInformation(cbct_sim)
    sitk.WriteImage(
        npimg,
        "/home/crohling/Documents/MC-GPU_v1.3_RELEASE_2012-12-12/Simulation/bin_00_2/recon_flipped_ohne.mha",
    )


if __name__ == "__main__":
    # analyzeCat()
    # getNPhoton(para)
    # analyzePatient()
    # flipRecon()
    im = np.load("/home/crohling/amalthea/data/results/low_022_test3/patient_22_proj:0")
    plt.imshow(im)
    plt.show()
