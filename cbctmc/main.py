import os
import SimpleITK as sitk
import numpy as np
import click
import pickle
from tqdm import tqdm
import multiprocessing
import shutil


def segMapToBinary(name):
    # Maps segmentation labels from "Total Segmentator" to an integer
    arr = ["spleen",
           "kidney_right",
           "kidney_left",
           "gallbladder",
           "liver",
           "stomach",
           "aorta",
           "inferior_vena_cava",
           "portal_vein_and_splenic_vein",
           "pancreas",
           "adrenal_gland_right",
           "adrenal_gland_left",
           "lung_upper_lobe_left",
           "lung_lower_lobe_left",
           "lung_upper_lobe_right",
           "lung_middle_lobe_right",
           "lung_lower_lobe_right",
           "vertebrae_L5",
           "vertebrae_L4",
           "vertebrae_L3",
           "vertebrae_L2",
           "vertebrae_L1",
           "vertebrae_T12",
           "vertebrae_T11",
           "vertebrae_T10",
           "vertebrae_T9",
           "vertebrae_T8",
           "vertebrae_T7",
           "vertebrae_T6",
           "vertebrae_T5",
           "vertebrae_T4",
           "vertebrae_T3",
           "vertebrae_T2",
           "vertebrae_T1",
           "vertebrae_C7",
           "vertebrae_C6",
           "vertebrae_C5",
           "vertebrae_C4",
           "vertebrae_C3",
           "vertebrae_C2",
           "vertebrae_C1",
           "esophagus",
           "trachea",
           "heart_myocardium",
           "heart_atrium_left",
           "heart_ventricle_left",
           "heart_atrium_right",
           "heart_ventricle_right",
           "pulmonary_artery",
           "brain",
           "iliac_artery_left",
           "iliac_artery_right",
           "iliac_vena_left",
           "iliac_vena_right",
           "small_bowel",
           "duodenum",
           "colon",
           "rib_left_1",
           "rib_left_2",
           "rib_left_3",
           "rib_left_4",
           "rib_left_5",
           "rib_left_6",
           "rib_left_7",
           "rib_left_8",
           "rib_left_9",
           "rib_left_10",
           "rib_left_11",
           "rib_left_12",
           "rib_right_1",
           "rib_right_2",
           "rib_right_3",
           "rib_right_4",
           "rib_right_5",
           "rib_right_6",
           "rib_right_7",
           "rib_right_8",
           "rib_right_9",
           "rib_right_10",
           "rib_right_11",
           "rib_right_12",
           "humerus_left",
           "humerus_right",
           "scapula_left",
           "scapula_right",
           "clavicula_left",
           "clavicula_right",
           "femur_left",
           "femur_right",
           "hip_left",
           "hip_right",
           "sacrum",
           "face",
           "gluteus_maximus_left",
           "gluteus_maximus_right",
           "gluteus_medius_left",
           "gluteus_medius_right",
           "gluteus_minimus_left",
           "gluteus_minimus_right",
           "autochthon_left",
           "autochthon_right",
           "iliopsoas_left",
           "iliopsoas_right",
           "urinary_bladder"]
    i = 1
    for names in arr:
        if names == name:
            return i
        i += 1
    print("Error: SegmentationFileName not known")
    return 1


def groupSegmentation(path):
    # Groups all segmentations from TotalSegmentator in one File,
    # maps segmentation labels to binary according to segMapToBinary()
    directory = os.fsencode(path)
    info_image = sitk.ReadImage(path + "/" + os.fsdecode(os.listdir(directory)[0]))
    size = info_image.GetSize()
    arr = np.zeros((size[2], size[1], size[0]))
    for file in tqdm(os.listdir(directory)):
        filename = os.fsdecode(file)
        img_ni = sitk.ReadImage(path + "/" + filename)
        img_np = sitk.GetArrayFromImage(img_ni)
        # img_np = np.flip(img_np, 2)
        arr = arr + img_np * segMapToBinary(filename.replace(".nii.gz", ""))
        os.remove(path + "/" + filename)
    total_seg = sitk.GetImageFromArray(arr)
    total_seg.CopyInformation(info_image)
    return total_seg


def createSegmentation(path, seg_filename, path_ct_in, filename_ct_in, gpu_id: int = 0):
    # Starts Total Segmentator with given CT Data
    file_path = path + "/" + seg_filename
    # remove old segmentation
    if os.path.exists(file_path):
        os.remove(file_path)
    # run segmentation (Look TotalSegmentator for details; URL: https://arxiv.org/abs/2208.05868.  arXiv: 2208.05868)
    os.system(f'CUDA_VISIBLE_DEVICES={gpu_id} TotalSegmentator -i ' + path_ct_in + "/" + filename_ct_in + " -o " + path)
    # bring segmentation in desired format
    sitk.WriteImage(groupSegmentation(path), file_path)


def segToMatFileNo(i, bar):
    # maps Segmentation integers from segMapToBinary() on integeres representing Materials in the Input file
    bar.update(1)
    if 1 <= i <= 4 or i == 10:
        return 3  # 'organs'
    elif 7 <= i <= 9 or i == 49 or 51 <= i <= 54:
        return 7  # 'blood'
    elif 11 <= i <= 12:
        return 10  # 'gland'
    elif 13 <= i <= 17:
        return 9  # 'lung'
    elif 55 <= i <= 57 or i == 6 or i == 42:
        return 14  # 'stomach'
    elif i == 5:
        return 13  # 'liver'
    elif 18 <= i <= 41 or 58 <= i <= 92:
        return 4  # 'bone'
    elif 94 <= i <= 103 or 44 <= i <= 48:
        return 2  # 'muscle'
    elif i == 43:
        return 0  # 'Luftroehre'
    elif i == 0:
        return -1  # 'unknown'
    else:
        return 99  # "error"


def getMaterial(hu, mat):
    if hu < -900:
        return "1 0.0012"
    elif -900 <= hu < -250:
        if mat == 9:
            return "9 0.382"
        else:
            return "1 0.0012"
    elif -250 <= hu < -20:
        if mat == 3:
            return "3 1.0"
        elif mat == 7:
            return "7 1.06"
        elif mat == 10 or mat == 9:
            return "10 1.03"
        elif mat == 14:
            return "14 1.04"
        elif mat == 13:
            return "13 1.05"
        elif mat == 2:
            return "2 1.05"
        elif mat == -1:
            return "6 0.95"
        elif mat == 4:
            return "12 1.03"
        else:
            return "6 0.95"
    elif -20 <= hu < 100:
        if mat == 3:
            return "3 1.0"
        elif mat == 7:
            return "7 1.06"
        elif mat == 10:
            return "10 1.03"
        elif mat == 14:
            return "14 1.04"
        elif mat == 13:
            return "13 1.05"
        elif mat == 2 or mat == -1:
            return "2 1.05"
        elif mat == 4:
            return "12 1.03"
        else:
            return "2 1.05"
    elif 100 <= hu < 300:
        if mat == 4 or mat == -1:
            return "5 1.1"
        elif mat == 2:
            return "2 1.05"
        if mat == 3:
            return "3 1.0"
        elif mat == 7:
            return "7 1.06"
        elif mat == 10:
            return "10 1.03"
        elif mat == 14:
            return "14 1.04"
        elif mat == 13:
            return "13 1.05"
        elif mat == 2 or mat == -1:
            return "2 1.05"
        else:
            return "2 1.05"
    elif 300 <= hu:
        return "4 1.99"


def writeVoxel(img, img_np, img_seg, path, path_out, vox_filename, vox_air_filename):
    print("Creating Voxel file")
    voxel_path = path + "/" + vox_filename
    voxel_air_path = path + "/" + vox_air_filename
    # CREATE VOXEL FILE, f: Voxel file representing patient, air: only air for calibration
    header = ""
    # WRITE VOXEL HEADER
    header += "[SECTION VOXELS HEADER v.2008-04-13]" + "\n"
    air_header = header
    # WRITE No. OF VOXELS IN X,Y,Z
    header += str(img.GetSize()[0]) + " " + str(img.GetSize()[1]) + " " + str(img.GetSize()[2]) + "\n"
    air_header += "1 1 1\n"
    # WRITE VOXEL SIZE (cm) ALONG X,Y,Z
    air_header += (str(img.GetSpacing()[0] / 10 * img.GetSize()[0]) + " " +
                   str(img.GetSpacing()[1] / 10 * img.GetSize()[1]) + " " +
                   str(img.GetSpacing()[2] / 10 * img.GetSize()[2]) + "\n")
    header += str(img.GetSpacing()[0] / 10) + " " + str(img.GetSpacing()[1] / 10) + " " + str(
        img.GetSpacing()[2] / 10) + "\n"
    # WRITE COLUMN NUMBER WHERE MATERIAL ID IS LOCATED
    header += str(1) + "\n"
    air_header += str(1) + "\n"
    # WRITE COLUMN NUMBER WHERE MASS DENSITY IS LOCATED
    header += str(2) + "\n"
    air_header += str(2) + "\n"
    # BLANK LINES AT END OF X,Y-CYCLES (1=YES,0=NO)
    header += str(0) + "\n"
    air_header += str(1) + "\n"
    # WRITE END HEADER
    header += "[END OF VXH SECTION]" + "\n"
    air_header += "[END OF VXH SECTION]" + "\n"

    img_np = np.transpose(img_np, (1, 2, 0))

    img_seg_np = np.transpose(sitk.GetArrayFromImage(img_seg), (1, 2, 0))
    func_segToMat = np.vectorize(segToMatFileNo)
    func_getMat = np.vectorize(getMaterial)
    print(img_np.shape)
    with tqdm(total=img_np.shape[0]*img_np.shape[1]*img_np.shape[2]) as bar:
        vox = func_getMat(img_np, func_segToMat(img_seg_np, bar))
        # Only for calibration with CatPhan 604, vox_phantom.npy is Voxel representation of CatPhan 604, notice:
        # different Material Files have to be set, if working with the Cat Phan; see: writeInputFile()
        # vox = np.load("vox_phantom.npy")
        # vox = np.swapaxes(vox,0,1)
    print(vox.shape)

    vox_seg = vox.astype("<U2").astype(float)
    vox_seg = np.transpose(vox_seg, (2, 0, 1))
    vox_seg = sitk.GetImageFromArray(vox_seg)
    vox_seg.CopyInformation(img)
    sitk.WriteImage(vox_seg, path_out + "/VoxSeg.mha")

    with open(voxel_path, 'w') as outfile:
        outfile.write(header)
        for i in tqdm(range(vox.shape[2])):
            np.savetxt(outfile, vox[:, :, i], delimiter='\n', newline='\n\n', fmt="%s10")
            outfile.write('\n')
        outfile.close()

    air = open(voxel_air_path, "w")
    air.write(air_header)
    air.write("1 0.0012")
    air.close()


def printMeta(img):
    # only for debugging purposes
    for k in img.GetMetaDataKeys():
        v = img.GetMetaData(k)
        print(f'({k}) = = "{v}"')
    print("Origin = ", img.GetOrigin())
    print("Spacing = ", img.GetSpacing())
    print("Dimensions = ", img.GetDimension())
    print("Size = ", img.GetSize())


def npToNifti(path, process_path, out_filename, np_filename, np_air_filename, spacing, normalize: bool = True,
              combine_photons: bool=True):
    # transforms numpy files of the simulation to standard CBCT data format, normalizes Data according to X-Ray
    # absorption law
    print("Converting Projetion data to nifti file format")
    with open(process_path + "/" + np_filename, 'rb') as f:
        proj = np.load(f)
    with open(process_path + "/" + np_air_filename, 'rb') as f:
        air = np.load(f)

    if normalize and combine_photons:
        # set half of minimal detection amplitude to every zero energy detection in order to avoid 0 division
        proj = np.where(proj == 0, 0.5 * np.min(proj[np.nonzero(proj)]), proj)
        proj = np.log(air / proj)  # normalize according to x-ray absorption law

    if not combine_photons:
        for i in range(4):
            proj_im = sitk.GetImageFromArray(proj[:, :, :, i])
            proj_im.SetSpacing((spacing, spacing, 1))
            proj_im.SetOrigin((int(-proj_im.GetSize()[0] * proj_im.GetSpacing()[0] / 2),
                               int(-proj_im.GetSize()[1] * proj_im.GetSpacing()[1] / 2), 1))
            sitk.WriteImage(proj_im, path + "/" + str(i) + "._" + out_filename)
    else:
        proj_im = sitk.GetImageFromArray(proj)
        proj_im.SetSpacing((spacing, spacing, 1))
        proj_im.SetOrigin((int(-proj_im.GetSize()[0] * proj_im.GetSpacing()[0] / 2),
                           int(-proj_im.GetSize()[1] * proj_im.GetSpacing()[1] / 2), 1))
        sitk.WriteImage(proj_im, path + "/" + out_filename)


def readDoseImage(filepath, det_pixel_y, det_pixel_x, combine_photons: bool = True):
    # reads data, adds up nonscattered and scattered photon energy counts and cuts detector image for artificial half
    # fan scan

    detenergy = np.loadtxt(filepath, dtype="float")
    if combine_photons:
        detenergy = detenergy[:, 0] + detenergy[:, 1] + detenergy[:, 2] + detenergy[:, 3]
        detenergy = np.reshape(detenergy, (int(detenergy.size / det_pixel_y), -1))
        detenergy = detenergy[:, 0:det_pixel_x]
    else:

        detenergy = np.reshape(detenergy, (int(detenergy[:,0].size / det_pixel_y), -1, 4))
        detenergy = detenergy[:, 0:det_pixel_x, :]
    detenergy = np.flip(detenergy, 0)
    return detenergy


def createNumpy(path, np_filename, np_air_filename, sim_path, sim_filename, sim_air_filename,
                no_sim, det_pixel_x_halffan, det_pixel_x, combine_photons: bool = True):
    # reads MCGPU output and returns all projections in one Numpy file
    print("Read projection data")
    items = []
    proj = []
    for i in range(no_sim):
        dat = "000" + str(i)
        dat = "_" + str(dat[-4:])
        if no_sim == 1:
            dat = ""
        items.append((sim_path + "/" + sim_filename + dat, det_pixel_x_halffan, det_pixel_x, combine_photons))
    with multiprocessing.Pool() as pool:
        for results in pool.starmap(readDoseImage, tqdm(items)):
            proj.append(results)
    proj = np.array(proj)
    air = readDoseImage(sim_path + "/" + sim_air_filename, det_pixel_x_halffan, det_pixel_x)
    shutil.rmtree(sim_path)
    with open(path + "/" + np_filename, 'wb') as f:
        np.savez_compressed(f, proj)
    with open(path + "/" + np_air_filename, 'wb') as f:
        np.save(f, air)


def writeXML(path, geo_filename, src_to_iso, src_to_det, no, lat_displacement):
    # write Geometry file for image reconstruction according scan parameters the user defined
    start = 270
    step = 360 / no
    f = open(path + "/" + geo_filename, "w")
    f.write("<?xml version=\"1.0\"?>" + "\n" + "<!DOCTYPE RTKGEOMETRY>" + "\n"
            + "<RTKThreeDCircularGeometry version=\"3\">" + "\n")
    f.write("<SourceToIsocenterDistance>" + str(src_to_iso) + "</SourceToIsocenterDistance>" + "\n")
    f.write("<SourceToDetectorDistance>" + str(src_to_det) + "</SourceToDetectorDistance>" + "\n")
    for i in range(no):
        g_angle = start + i * step
        if g_angle > 360.0:
            g_angle = g_angle - 360
        g_angle_d = - np.deg2rad(g_angle)
        f.write("<Projection>" + "\n")
        f.write("<GantryAngle>" + str(g_angle) + "</GantryAngle>" + "\n")
        f.write("<ProjectionOffsetX>" + str(lat_displacement) + "</ProjectionOffsetX>" + "\n")
        f.write("<ProjectionOffsetY>0.0</ProjectionOffsetY>" + "\n")
        f.write("<Matrix>" + "\n")
        f.write(str(-np.cos(g_angle_d) * src_to_det + lat_displacement * np.sin(g_angle_d)) + "\t" + "0" + "\t"
                + str(-src_to_det * np.sin(g_angle_d) - lat_displacement * np.cos(g_angle_d)) + "\t" + str(
            lat_displacement * src_to_iso) + "\n"
                + "0" + "\t" + str(-src_to_det) + "\t" + "0" + "\t" + "0" + "\n"
                + str(-np.sin(g_angle_d)) + "\t" + "0" + "\t" + str(np.cos(g_angle_d)) + "\t"
                + str(-src_to_iso) + "\n")
        f.write("</Matrix>" + "\n")
        f.write("</Projection>" + "\n")
    f.write("</RTKThreeDCircularGeometry>")
    f.close()


def writeInputFile(path, filename, sim_filename, sim_air_filename, vox_filename,
                   vox_air_filename, in_filename, in_air_filename, ct_size, ct_spacing, photons,
                   src_to_iso, src_to_det, no, lat_displacement, det_pix_x, det_pix_y, det_pixel_size,
                   det_pix_x_halffan, air=False, random_seed=42):
    # writes input file according to the MC-GPU standard, commented material files are used for Catphan 604 calibration
    print("Writing input file")
    size_x = ct_size[0] * ct_spacing[0]
    size_y = ct_size[1] * ct_spacing[1]
    size_z = ct_size[2] * ct_spacing[2]
    if not air:
        f = open(path + "/" + in_filename, "w")
    else:
        f = open(path + "/" + in_air_filename, "w")
    text = ("# >>>> INPUT FILE FOR MC-GPU v1.3 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> \n"
            + "\n#[SECTION SIMULATION CONFIG v.2009-05-12]\n")

    text += str(photons) + " # TOTAL NUMBER OF HISTORIES, OR SIMULATION TIME IN SECONDS IF VALUE < 100000\n"

    text += (f"{random_seed}                      # RANDOM SEED (ranecu PRNG)\n0                               "
             + "# GPU NUMBER TO USE WHEN MPI IS NOT USED, OR TO BE AVOIDED IN MPI RUNS\n128         "
             + "# GPU THREADS PER CUDA BLOCK (multiple of 32)\n150                             "
             + "# SIMULATED HISTORIES PER GPU THREAD\n"
             + "\n#[SECTION SOURCE v.2011-07-12]\n" + "/3rd_party/mcgpu/125kVp_0.89mmTi.spc" + "     # X-RAY ENERGY SPECTRUM FILE\n"
             + str(float(size_x / 20)) + " " + str(float(size_y / 20) - src_to_iso / 10) + " " + str(
                float(size_z / 20)) + " "
             + "           # SOURCE POSITION: X Y Z [cm]\n"
             + "0.0   1.0   0.0                # SOURCE DIRECTION COSINES: U V W\n"
             + "-15.0 -15.0                     # POLAR AND AZIMUTHAL APERTURES FOR THE FAN BEAM [degrees]"
             + " (input negative to cover the whole detector)\n"
             + "\n#[SECTION IMAGE DETECTOR v.2009-12-02]\n" + "/output/")

    if not air:
        text += sim_filename + "           # OUTPUT IMAGE FILE NAME\n"
    else:
        text += sim_air_filename + "           # OUTPUT IMAGE FILE NAME\n"

    text += (str(det_pix_x_halffan) + " " + str(det_pix_y)
             + "                    # NUMBER OF PIXELS IN THE IMAGE: Nx Nz\n "
             + str((det_pix_x * det_pixel_size + 2 * np.abs(lat_displacement)) / 10)
             + " " + str(det_pix_y * det_pixel_size / 10)
             + "        # IMAGE SIZE (width, height): Dx Dz [cm]\n "
             + str(float(src_to_det / 10))
             + "   # SOURCE-TO-DETECTOR DISTANCE (detector set in front of the source,"
             + " perpendicular to the initial direction)\n"
             + "\n#[SECTION CT SCAN TRAJECTORY v.2011-10-25]\n")

    if not air:
        text += str(no)
    else:
        text += str(1)

    text += ("\t     # NUMBER OF PROJECTIONS (beam must be perpendicular to Z axis, set to 1 for a single projection)\n"
             + str(
                360 / no) + "                # ANGLE BETWEEN PROJECTIONS [degrees] (360/num_projections for full CT)\n"
             + "0.0 5000.0              # ANGLES OF INTEREST (projections outside the input interval will be skipped)\n"
             + str(
                src_to_iso / 10) + "       # SOURCE-TO-ROTATION AXIS DISTANCE (rotation radius, axis parallel to Z)\n "
             + "0.0                            # VERTICAL TRANSLATION BETWEEN PROJECTIONS (HELICAL SCAN)\n"
             + "\n#[SECTION DOSE DEPOSITION v.2012-12-12]\n"
             + "NO        # TALLY MATERIAL DOSE? [YES/NO] (electrons not transported,"
             + " x-ray energy locally deposited at interaction)\n"
             + "NO                        # TALLY 3D VOXEL DOSE? [YES/NO] (dose measured separately for each voxel)\n")

    if not air:
        text += "/output" + "/" + filename + "_dose.dat             # OUTPUT VOXEL DOSE FILE NAME"
    else:
        text += "/output" + "/" + filename + "air_dose.dat          # OUTPUT VOXEL DOSE FILE NAME"

    text += ("\n 1 " + str(1) + "           #Dose ROI X\n 1 " + str(1)
             + "       #Dose ROI Y\n 1 " + str(1) + "         #Dose ROI Z\n")

    if not air:
        text += "\n#[SECTION VOXELIZED GEOMETRY FILE v.2009-11-30]\n" + "/input/" + vox_filename
    else:
        text += "\n#[SECTION VOXELIZED GEOMETRY FILE v.2009-11-30]\n" + "/input/" + vox_air_filename

    text += ("\n\n#[SECTION MATERIAL FILE LIST v.2009-11-30]\n"
             + "/3rd_party/mcgpu/MC-GPU_material_files/air_new_5-125keV.mcgpu      #  1st MATERIAL FILE (.gz accepted)\n")

    if not air:
        text += ("/3rd_party/mcgpu/MC-GPU_material_files/muscle_tissue_new__5-125keV.mcgpu"
                 + "# 2nd MATERIAL FILE\n"
                 + "/3rd_party/mcgpu/MC-GPU_material_files/soft_tissue_new__5-125keV.mcgpu        "
                 + "# 3rd MATERIAL FILE\n"
                 + "/3rd_party/mcgpu/MC-GPU_material_files/bone_new__5-125keV.mcgpu                 "
                 + "#  4th MATERIAL FILE\n"
                 + "/3rd_party/mcgpu/MC-GPU_material_files/cartilage_new__5-125keV.mcgpu            "
                 + "#  5th MATERIAL FILE\n"
                 + "/3rd_party/mcgpu/MC-GPU_material_files/adipose_new__5-125keV.mcgpu              "
                 + "#  6th MATERIAL FILE\n"
                 + "/3rd_party/mcgpu/MC-GPU_material_files/blood_new__5-125keV.mcgpu                "
                 + "#  7th MATERIAL FILE\n"
                 + "/3rd_party/mcgpu/MC-GPU_material_files/H2O_new__5-125keV.mcgpu                 "
                 + "#  8th MATERIAL FILE\n"
                 + "/3rd_party/mcgpu/MC-GPU_material_files/lung_new__5-125keV.mcgpu                 "
                 + "#  9th MATERIAL FILE\n"
                 + "/3rd_party/mcgpu/MC-GPU_material_files/glands_others_new__5-125keV.mcgpu        "
                 + "# 10th MATERIAL FILE\n"
                 + "/3rd_party/mcgpu/MC-GPU_material_files/H2O_new__5-125keV.mcgpu                "
                 + "# 11th MATERIAL FILE\n"
                 + "/3rd_party/mcgpu/MC-GPU_material_files/red_marrow_new__5-125keV.mcgpu           "
                 + "# 12th MATERIAL FILE\n"
                 + "/3rd_party/mcgpu/MC-GPU_material_files/liver_new__5-125keV.mcgpu               "
                 + "# 13th MATERIAL FILE\n"
                 + "/3rd_party/mcgpu/MC-GPU_material_files/stomach_intestines_new__5-125keV.mcgpu   "
                 + "#  14th MATERIAL FILE\n"
                 + "/3rd_party/mcgpu/MC-GPU_material_files/H2O_new__5-125keV.mcgpu                     "
                 + "#  15th MATERIAL FILE\n\n\n ")
        # # material files created for calibration with CatPhan 604

        # text += (" /3rd_party/mcgpu/MC-GPU_material_files/PMP_new_5-125keV.mcgpu    #  2nd MATERIAL FILE\n"
        #           
        #          + " /3rd_party/mcgpu/MC-GPU_material_files/LDPE_new_5-125keV.mcgpu         "
        #          + "# 3rd MATERIAL FILE\n"  
        #          + " /3rd_party/mcgpu/MC-GPU_material_files/Polystrene_new_5-125keV.mcgpu                 "
        #          + "#  4th MATERIAL FILE\n"  
        #          + " /3rd_party/mcgpu/MC-GPU_material_files/Acryl_new_5-125keV.mcgpu           "
        #          + "#  5th MATERIAL FILE\n"  
        #          + " /3rd_party/mcgpu/MC-GPU_material_files/bone_20%_new_5-125keV.mcgpu              "
        #          + "#  6th MATERIAL FILE\n"  
        #          + " /3rd_party/mcgpu/MC-GPU_material_files/Delrin_new_5-125keV.mcgpu                "
        #          + "#  7th MATERIAL FILE\n"  
        #          + " /3rd_party/mcgpu/MC-GPU_material_files/Bone_50%_new_5-125keV.mcgpu                "
        #          + "#  8th MATERIAL FILE\n"  
        #          + " /3rd_party/mcgpu/MC-GPU_material_files/Teflon_new_5-125keV.mcgpu                 "
        #          + "#  9th MATERIAL FILE\n"  
        #          + " /3rd_party/mcgpu/MC-GPU_material_files/H2O_new_5-125keV.mcgpu                 "
        #          + "#  10th MATERIAL FILE\n\n\n ")

    text += "# >>>> END INPUT FILE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n"
    f.write(text)
    f.close()


def runSimulation(path, gpu_id: int = 0):
    os.chdir(path)
    os.system(f"docker run --rm --gpus device={gpu_id} -v $(pwd)/input:/input -v $(pwd)/output:/output -u $(id -u):$(id -g) "
              "imaging MC-GPU_v1.3.x /input/input.in")
    os.system(f"docker run --rm --gpus device={gpu_id} -v $(pwd)/input:/input -v $(pwd)/output:/output -u $(id -u):$(id -g) "
              "imaging MC-GPU_v1.3.x /input/input_air.in")

@click.command()
@click.option('--path_ct_in', help='Path to ct file')
@click.option('--filename_ct_in', help='CT filename')
@click.option('--path_out', help='Desired Output folder')
@click.option('--filename', help='Desired Output file name')
@click.option('--no_sim', default=894, help='Number of Projections.')
@click.option('--det_pix_size', default=0.776, help='Size of one Detector pixel in mm')
@click.option('--det_pix_x', default=512, help='Number of Detector-pixel in X-Direction')
@click.option('--det_pix_y', default=384, help='Number of Detector-pixel in Y-Direction')
@click.option('--lat_displacement', default=-160, help='If cbct is used in half fan mode, give lateral displacement')
@click.option('--src_to_detector', default=1500, help='Distance between X-Ray source and detector in mm')
@click.option('--src_to_iso', default=1000, help='Distance between X-Ray source and rotation center in mm')
@click.option('--photons', default=2.4e9, help='Number of photons to Simulate for each projection')
@click.option('--force_rerun', default=False, help='Set force_rerun=True to redo everything')
@click.option('--force_segment', default=False, help='Set force_segment=True to redo segmentation')
@click.option('--force_create_object', default=False, help='Set force_create_object=True to redo the object')
@click.option('--force_simulate', default=False, help='Set force_simulate=True to redo simulation')
@click.option('--gpu_id', default=0, type=click.INT, help='PCI ID of GPU for MC simulation')
@click.option('--random_seed', default=42, type=click.INT, help='Random seed for MC simulation')
@click.option('--normalize', default=True, type=click.BOOL, help='Set normalize = False to get Raw, unnormalized '
                                                                 'Simulation data')
@click.option('--combine_photons', default=True, type=click.BOOL, help='Set combine_photons = False to get 4 outputs,'
                                                                       'containing non -, compton - , rayleigh and '
                                                                       'multiple scattered photon detections. If set '
                                                                       'False, normalize is automatically set to False ')
def run(path_ct_in, filename_ct_in, path_out, filename, no_sim, det_pix_size,
         det_pix_x, det_pix_y, lat_displacement, src_to_detector, src_to_iso, photons, force_rerun, force_segment,
         force_create_object, force_simulate, gpu_id, random_seed, normalize, combine_photons):
    # #### Setup #############################################
    # create Files, define paths
    if not os.path.exists(path_out):
        os.makedirs(path_out)

    input_path = path_out + "/input"
    output_path = path_out + "/output"
    process_path = path_out + "/process"
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(process_path):
        os.makedirs(process_path)
    seg_path = path_ct_in + "/segmentation_" + filename_ct_in[:-4]
    seg_filename = filename_ct_in[:-4] + "_seg.nii"
    if not os.path.exists(seg_path):
        os.makedirs(seg_path)

    sim_path = output_path
    sim_filename = filename + "_image.dat"
    sim_air_filename = filename + "_air_image.dat"
    vox_filename = "geometry" + ".vox"
    vox_air_filename = "geometry" + "_air.vox"
    geo_filename = filename + ".xml"
    np_filename = filename + "_np.npy"
    np_air_filename = filename + "_air_np.npy"
    in_filename = "input.in"
    in_air_filename = "input_air.in"
    out_filename = "sim_" + filename + ".mha"
    log_filename = "log.pkl"


    # create sitk Images
    img_ct = sitk.ReadImage(path_ct_in + "/" + filename_ct_in)
    img_np = sitk.GetArrayFromImage(img_ct)
    # define Variables
    det_pix_x_halffan = int(det_pix_x + np.abs(lat_displacement) / det_pix_size * 2)
    ###########################################################

    # #### Run ###############################################
    # Use TotalSegmentator to segment ct
    if not os.path.exists(seg_path + "/" + seg_filename) or force_segment or force_rerun:
        createSegmentation(seg_path, seg_filename, path_ct_in, filename_ct_in, gpu_id=gpu_id)
    img_seg = sitk.ReadImage(seg_path + "/" + seg_filename)

    # Create Voxel Object
    if not os.path.exists(input_path + "/" + vox_filename) or force_create_object or force_rerun:
        writeVoxel(img_ct, img_np, img_seg, input_path, path_out, vox_filename, vox_air_filename)

    # Prepare and Run MC-GPU Simulation
    if not os.path.exists(process_path + "/" + log_filename) or force_simulate or force_rerun:
        writeInputFile(input_path, filename, sim_filename, sim_air_filename, vox_filename,
                       vox_air_filename, in_filename, in_air_filename, img_ct.GetSize(), img_ct.GetSpacing(), photons,
                       src_to_iso, src_to_detector, no_sim, lat_displacement, det_pix_x,
                       det_pix_y, det_pix_size, det_pix_x_halffan, random_seed=random_seed)
        writeInputFile(input_path, filename, sim_filename, sim_air_filename, vox_filename,
                       vox_air_filename, in_filename, in_air_filename, img_ct.GetSize(), img_ct.GetSpacing(), photons,
                       src_to_iso, src_to_detector, no_sim, lat_displacement, det_pix_x,
                       det_pix_y, det_pix_size, det_pix_x_halffan, air=True, random_seed=random_seed)
        runSimulation(path_out, gpu_id=gpu_id)
        # create log file
        with open(process_path + "/" + log_filename, 'wb') as f:
            pickle.dump([no_sim, det_pix_size, det_pix_x, det_pix_y, lat_displacement,
                         src_to_detector, src_to_iso, photons], f)

    # read log file
    with open(process_path + "/" + log_filename, "rb") as f:
        no_sim, det_pix_size, det_pix_x, det_pix_y, lat_displacement, src_to_detector, \
         src_to_iso, photons = pickle.load(f)

    # bring simulation to SimpleITK format and write Geometry file
    createNumpy(process_path, np_filename, np_air_filename, sim_path, sim_filename, sim_air_filename, no_sim,
                det_pix_x_halffan, det_pix_x, combine_photons=combine_photons)
    npToNifti(path_out, process_path, out_filename, np_filename, np_air_filename, det_pix_size, normalize=normalize,
              combine_photons=combine_photons)
    writeXML(path_out, geo_filename, src_to_iso, src_to_detector, no_sim, lat_displacement)
    #############################################################


if __name__ == '__main__':
    run()

