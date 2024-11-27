pixel_area = 0.006024
mean_energy = 62889.36670284205
factor_beam_hardening = 1.09
n_photons_low = 5e7
n_photons_high = 2.4e9
scale_low_fit = 4.398481863563472
scale_high_fit = 212.8556560184687
scale_low_theo = pixel_area / (mean_energy * factor_beam_hardening) * n_photons_low
scale_high_theo = pixel_area / (mean_energy * factor_beam_hardening) * n_photons_high


# detector pixel stats
# maximum of max and mean of p99
low_photon_stats = {"max": 8.711442, "p99": 4.340285567627859}
high_photon_stats = {"max": 7.217965, "p99": 4.320198940555545}
global_max_pixel_value = max(low_photon_stats["max"], high_photon_stats["max"])


HIGH_VAR_MEAN_RATIO = 0.00477792
