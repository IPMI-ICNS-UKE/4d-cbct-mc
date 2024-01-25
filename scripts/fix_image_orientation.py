import SimpleITK as sitk

direction_rai = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

# for i_bin in range(0, 10):
#     print(i_bin)
#     image = sitk.ReadImage(
#         f"/mnt/nas_io/anarchy/4d_cbct_mc/4d/R2017025/ct/bin_{i_bin:02d}.nii"
#     )
#     direction = image.GetDirection()
#
#     x_rai_direction = direction_rai[0:3]
#     y_rai_direction = direction_rai[3:6]
#     z_rai_direction = direction_rai[6:9]
#
#     x_direction = direction[0:3]
#     y_direction = direction[3:6]
#     z_direction = direction[6:9]
#
#     if x_direction != x_rai_direction:
#         # flip x
#         image = sitk.Flip(image, [True, False, False])
#
#     if y_direction != y_rai_direction:
#         # flip y
#         image = sitk.Flip(image, [False, True, False])
#
#     if z_direction != z_rai_direction:
#         # flip z
#         image = sitk.Flip(image, [False, False, True])
#
#     image.SetDirection(direction_rai)
#     sitk.WriteImage(
#         image, f"/mnt/nas_io/anarchy/4d_cbct_mc/4d/R2017025/ct_rai/bin_{i_bin:02d}.nii"
#     )


image = sitk.ReadImage(f"/mnt/nas_io/anarchy/4d_cbct_mc/4d/R2017025/ct/avg.nii")
direction = image.GetDirection()

x_rai_direction = direction_rai[0:3]
y_rai_direction = direction_rai[3:6]
z_rai_direction = direction_rai[6:9]

x_direction = direction[0:3]
y_direction = direction[3:6]
z_direction = direction[6:9]

if x_direction != x_rai_direction:
    # flip x
    image = sitk.Flip(image, [True, False, False])

if y_direction != y_rai_direction:
    # flip y
    image = sitk.Flip(image, [False, True, False])

if z_direction != z_rai_direction:
    # flip z
    image = sitk.Flip(image, [False, False, True])

image.SetDirection(direction_rai)
sitk.WriteImage(image, f"/mnt/nas_io/anarchy/4d_cbct_mc/4d/R2017025/ct_rai/avg.nii")
