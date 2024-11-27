import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


def pn():
    phantom_img = sitk.ReadImage(
        "/home/crohling/Documents/MC-GPU_v1.3_RELEASE_2012-12-12/Simulation/catphan_06/sim_phantom.mha"
    )
    phantom = sitk.GetArrayFromImage(phantom_img)
    for i in range(5):
        i = i + 2
        img = sitk.GetImageFromArray(phantom**i)
        img.CopyInformation(phantom_img)
        sitk.WriteImage(
            img,
            "/home/crohling/Documents/MC-GPU_v1.3_RELEASE_2012-12-12/Simulation/phantom_n/sim_phantom"
            + str(i)
            + ".mha",
        )


def createW():
    w = np.ones((256, 256))
    for x in range(256):
        for y in range(256):
            if 98 < np.sqrt((x - 128) ** 2 + (y - 128) ** 2) < 102:
                w[x, y] = 0
    np.save(
        "/home/crohling/Documents/MC-GPU_v1.3_RELEASE_2012-12-12/Simulation/phantom_n/w.npy",
        w,
    )


def createT():
    t = np.ones((256, 256)) * 0.004242744065194304
    for x in range(256):
        for y in range(256):
            if np.sqrt((x - 128) ** 2 + (y - 128) ** 2) < 100:
                t[x, y] = 0.02004867443846947
    np.save(
        "/home/crohling/Documents/MC-GPU_v1.3_RELEASE_2012-12-12/Simulation/phantom_n/t.npy",
        t,
    )


def run(no):
    f = np.empty((no, 256, 256))
    fall = np.empty((7, 256, 256, 256))
    f0 = sitk.ReadImage(
        "/home/crohling/Documents/MC-GPU_v1.3_RELEASE_2012-12-12/Simulation/phantom_n/phan_recon0.mha"
    )
    fall[0] = sitk.GetArrayFromImage(f0)
    f1 = sitk.ReadImage(
        "/home/crohling/Documents/MC-GPU_v1.3_RELEASE_2012-12-12/Simulation/catphan_06/recon.mha"
    )
    fall[1] = sitk.GetArrayFromImage(f1)
    f2 = sitk.ReadImage(
        "/home/crohling/Documents/MC-GPU_v1.3_RELEASE_2012-12-12/Simulation/phantom_n/phan_recon2.mha"
    )
    fall[2] = sitk.GetArrayFromImage(f2)
    f3 = sitk.ReadImage(
        "/home/crohling/Documents/MC-GPU_v1.3_RELEASE_2012-12-12/Simulation/phantom_n/phan_recon3.mha"
    )
    fall[3] = sitk.GetArrayFromImage(f3)
    f4 = sitk.ReadImage(
        "/home/crohling/Documents/MC-GPU_v1.3_RELEASE_2012-12-12/Simulation/phantom_n/phan_recon4.mha"
    )
    fall[4] = sitk.GetArrayFromImage(f4)
    f5 = sitk.ReadImage(
        "/home/crohling/Documents/MC-GPU_v1.3_RELEASE_2012-12-12/Simulation/phantom_n/phan_recon5.mha"
    )
    fall[5] = sitk.GetArrayFromImage(f5)
    f6 = sitk.ReadImage(
        "/home/crohling/Documents/MC-GPU_v1.3_RELEASE_2012-12-12/Simulation/phantom_n/phan_recon6.mha"
    )
    fall[6] = sitk.GetArrayFromImage(f6)

    for i in range(50):
        i += 130
        for j in range(no):
            f[j] += fall[j, :, i, :]
    f = f / 50
    w = np.load(
        "/home/crohling/Documents/MC-GPU_v1.3_RELEASE_2012-12-12/Simulation/phantom_n/w.npy"
    )
    t = np.load(
        "/home/crohling/Documents/MC-GPU_v1.3_RELEASE_2012-12-12/Simulation/phantom_n/t.npy"
    )

    plt.imshow(w)
    plt.show()
    plt.imshow(t)
    plt.show()
    for i in range(no):
        plt.imshow(f[i])
        plt.show()
    b = np.empty((no, no))
    a = np.empty(no)
    for i in range(no):
        for j in range(no):
            b[i, j] = np.sum(w * f[i] * f[j])
        a[i] = np.sum(w * f[i] * t)
    binv = np.linalg.inv(b)
    c = binv.dot(a)
    final = np.zeros((256, 256))
    for i in range(no):
        final += c[i] * f[i]
    plt.imshow(final)
    plt.show()
    plt.imshow(final)
    plt.show()
    print(c)


if __name__ == "__main__":
    createT()
    # createW()
    run(5)
