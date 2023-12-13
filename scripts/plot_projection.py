from matplotlib import pyplot as plt

from cbctmc.mc.projection import MCProjection

p_full = MCProjection.from_file(
    "/datalake_fast/mc_test/mc_output/air_halffan_test/out_fullfan/projection",
    n_detector_pixels=(924 * 2, 384 * 2),
    n_detector_pixels_half_fan=(1024, 768),
)


n_projections = 10
fig, ax = plt.subplots(1, n_projections + 1, sharex=True, sharey=True)
ax[0].imshow(p_full[..., 0])
for i in range(n_projections):
    p_half = MCProjection.from_file(
        f"/datalake_fast/mc_test/mc_output/air_halffan_test/out_halffan/projection_{i:04d}",
        n_detector_pixels=(1848, 768),
        n_detector_pixels_half_fan=(1024, 768),
    )
    ax[i + 1].imshow(p_half[..., 0])
