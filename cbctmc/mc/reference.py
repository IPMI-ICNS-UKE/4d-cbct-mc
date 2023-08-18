# reference attenuation coefficients for mean spectrum energy of 63.140 keV in 1/mm
REFERENCE_MU = {
    "air": 0.000023674711138187246,
    "h2o": 0.020119709288519042,
    "teflon": 0.03943393182174662,
    "bone_050": 0.03480381262984748,
    "bone_020": 0.024925935187940915,
    "delrin": 0.02694022154936656,
    "acrylic": 0.022290157393600557,
    "polystyrene": 0.01896977750638363,
    "ldpe": 0.017862982216811124,
    "pmp": 0.016115516565166557,
}


# mu values extracted from real CatPhan604 Varian Scan
# these may not match with the real REFRENCE_MU above, especially for higher density
# materials (e.g. bone, teflon, etc.)
REFERENCE_MU_VARIAN = {
    "h2o": 0.0204,
    "air": 0.00423945385,  # mean of the two air ROIs
    "teflon": 0.03372094,
    "bone_050": 0.030424252,
    "bone_020": 0.023067258,
    "delrin": 0.024775395,
    "acrylic": 0.021296123,
    "polystyrene": 0.018962856,
    "ldpe": 0.018118449,
    "pmp": 0.016767636,
}
