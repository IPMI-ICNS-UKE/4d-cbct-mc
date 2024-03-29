import xraydb

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

REFERENCE_MU_75KEV = {
    "air": xraydb.material_mu("air", 75.0 * 10**3) / 10.0,
    "h2o": xraydb.material_mu("water", 75.0 * 10**3) / 10.0,
    "teflon": xraydb.material_mu("CF2", 75.0 * 10**3, density=2.16) / 10.0,
    "bone_050": xraydb.material_mu(
        "C0.36Ca0.14H0.04N0.06O0.34P0.06", 75.0 * 10**3, density=1.4
    )
    / 10.0,
    "bone_020": xraydb.material_mu(
        "C0.49Ca0.06H0.06N0.06O0.3P0.03", 75.0 * 10**3, density=1.14
    )
    / 10.0,
    "delrin": xraydb.material_mu("CH2O", 75.0 * 10**3, density=1.42) / 10.0,
    "acrylic": xraydb.material_mu("C5H8O2", 75.0 * 10**3, density=1.18) / 10.0,
    "polystyrene": xraydb.material_mu("C8H8", 75.0 * 10**3, density=1.03) / 10.0,
    "ldpe": xraydb.material_mu("C2H4", 75.0 * 10**3, density=0.92) / 10.0,
    "pmp": xraydb.material_mu("C6H12", 75.0 * 10**3, density=0.83) / 10.0,
}

# mu values extracted from real CatPhan604 Varian Scan
# these may not match with the real REFRENCE_MU above, especially for higher density
# materials (e.g. bone, teflon, etc.)
# REFERENCE_MU_VARIAN = {
#     "h2o": 0.0204,
#     "air": 0.00423945385,  # mean of the two air ROIs
#     "teflon": 0.03372094,
#     "bone_050": 0.030424252,
#     "bone_020": 0.023067258,
#     "delrin": 0.024775395,
#     "acrylic": 0.021296123,
#     "polystyrene": 0.018962856,
#     "ldpe": 0.018118449,
#     "pmp": 0.016767636,
# }

REFERENCE_MU_VARIAN = {
    "h2o": 0.0204,
    "air": 0.004239453934133053,  # mean of the two air ROIs
    "air_1": 0.00420496566221118,
    "teflon": 0.033720940351486206,
    "delrin": 0.024775395169854164,
    "bone_020": 0.023067258298397064,
    "acrylic": 0.021296123042702675,
    "air_2": 0.004273942206054926,
    "polystyrene": 0.018962856382131577,
    "ldpe": 0.018118448555469513,
    "bone_050": 0.030424252152442932,
    "pmp": 0.016767635941505432,
    "water": 0.020360935479402542,
}


OLD_REFERENCE_ROI_STATS_CATPHAN604_VARIAN = {
    "acrylic": {
        "evaluated_voxels": 702,
        "max": 0.024565162,
        "mean": 0.02118984,
        "min": 0.01843044,
        "p25": 0.020543355029076338,
        "p50": 0.021203051321208477,
        "p75": 0.021891893818974495,
        "std": 0.0010002705,
    },
    "air_1": {
        "evaluated_voxels": 738,
        "max": 0.0069951597,
        "mean": 0.0042614317,
        "min": 0.0017914834,
        "p25": 0.0036453650100156665,
        "p50": 0.0043046504724770784,
        "p75": 0.004874854581430554,
        "std": 0.0008872237,
    },
    "air_2": {
        "evaluated_voxels": 738,
        "max": 0.007203701,
        "mean": 0.004263839,
        "min": 0.0007208627,
        "p25": 0.0035895351320505142,
        "p50": 0.004292335128411651,
        "p75": 0.0049290425376966596,
        "std": 0.00093796407,
    },
    "bone_020": {
        "evaluated_voxels": 702,
        "max": 0.026355004,
        "mean": 0.023010395,
        "min": 0.01974901,
        "p25": 0.022336481139063835,
        "p50": 0.023006029427051544,
        "p75": 0.023627137299627066,
        "std": 0.0009935343,
    },
    "bone_050": {
        "evaluated_voxels": 702,
        "max": 0.03361125,
        "mean": 0.03033594,
        "min": 0.02685419,
        "p25": 0.029607506934553385,
        "p50": 0.03038214612752199,
        "p75": 0.031094387639313936,
        "std": 0.0011011023,
    },
    "delrin": {
        "evaluated_voxels": 738,
        "max": 0.028028259,
        "mean": 0.024534198,
        "min": 0.019162796,
        "p25": 0.023890687618404627,
        "p50": 0.02462920267134905,
        "p75": 0.02532502356916666,
        "std": 0.001283905,
    },
    "ldpe": {
        "evaluated_voxels": 738,
        "max": 0.02204132,
        "mean": 0.018294167,
        "min": 0.01422185,
        "p25": 0.01753551885485649,
        "p50": 0.018273623660206795,
        "p75": 0.01908028405159712,
        "std": 0.0011521464,
    },
    "pmp": {
        "evaluated_voxels": 702,
        "max": 0.022801593,
        "mean": 0.017026626,
        "min": 0.013592942,
        "p25": 0.016201759222894907,
        "p50": 0.01691317930817604,
        "p75": 0.017683714162558317,
        "std": 0.0012761452,
    },
    "polystyrene": {
        "evaluated_voxels": 702,
        "max": 0.022057742,
        "mean": 0.018940382,
        "min": 0.014543693,
        "p25": 0.01829702267423272,
        "p50": 0.018941940739750862,
        "p75": 0.019586036913096905,
        "std": 0.0009991035,
    },
    "teflon": {
        "evaluated_voxels": 702,
        "max": 0.036407672,
        "mean": 0.032669637,
        "min": 0.01873422,
        "p25": 0.03252708353102207,
        "p50": 0.03334852121770382,
        "p75": 0.03420156892389059,
        "std": 0.0031883258,
    },
}

REFERENCE_ROI_STATS_CATPHAN604_VARIAN = {
    "air_1": {
        "min": 0.0018588077509775758,
        "max": 0.006995159666985273,
        "mean": 0.004297331906855106,
        "p25": 0.0037020157906226814,
        "p50": 0.004334207391366363,
        "p75": 0.004905232577584684,
        "std": 0.0008914025384001434,
        "evaluated_voxels": 672,
    },
    "teflon": {
        "min": 0.02998066321015358,
        "max": 0.0372336246073246,
        "mean": 0.03361523896455765,
        "p25": 0.03295155335217714,
        "p50": 0.033612070605158806,
        "p75": 0.03434935491532087,
        "std": 0.0010753646492958069,
        "evaluated_voxels": 714,
    },
    "delrin": {
        "min": 0.021724404767155647,
        "max": 0.028028259053826332,
        "mean": 0.02472609281539917,
        "p25": 0.024046272039413452,
        "p50": 0.02470720000565052,
        "p75": 0.02541164169088006,
        "std": 0.0010216617956757545,
        "evaluated_voxels": 672,
    },
    "bone_020": {
        "min": 0.02039269730448723,
        "max": 0.02635500393807888,
        "mean": 0.023070329800248146,
        "p25": 0.022366449236869812,
        "p50": 0.023047080263495445,
        "p75": 0.02370102982968092,
        "std": 0.0010106356348842382,
        "evaluated_voxels": 714,
    },
    "acrylic": {
        "min": 0.01843043975532055,
        "max": 0.024565162137150764,
        "mean": 0.02121036686003208,
        "p25": 0.02057168073952198,
        "p50": 0.021216188557446003,
        "p75": 0.021898872684687376,
        "std": 0.0010135178454220295,
        "evaluated_voxels": 714,
    },
    "air_2": {
        "min": 0.0007208627066574991,
        "max": 0.007203700952231884,
        "mean": 0.00426891166716814,
        "p25": 0.0036289443960413337,
        "p50": 0.004292335128411651,
        "p75": 0.004904001136310399,
        "std": 0.0009401424322277308,
        "evaluated_voxels": 672,
    },
    "polystyrene": {
        "min": 0.014543692581355572,
        "max": 0.02205774188041687,
        "mean": 0.018922727555036545,
        "p25": 0.018277317751199007,
        "p50": 0.018946046009659767,
        "p75": 0.01951624872162938,
        "std": 0.0009755354840308428,
        "evaluated_voxels": 714,
    },
    "ldpe": {
        "min": 0.014221849851310253,
        "max": 0.021722761914134026,
        "mean": 0.018143903464078903,
        "p25": 0.01742673246189952,
        "p50": 0.018127480521798134,
        "p75": 0.018831512425094843,
        "std": 0.001071136794053018,
        "evaluated_voxels": 672,
    },
    "bone_050": {
        "min": 0.02685418911278248,
        "max": 0.033611249178647995,
        "mean": 0.030341893434524536,
        "p25": 0.029607506934553385,
        "p50": 0.0303566949442029,
        "p75": 0.031082894187420607,
        "std": 0.001093234634026885,
        "evaluated_voxels": 714,
    },
    "pmp": {
        "min": 0.01359294168651104,
        "max": 0.019478071480989456,
        "mean": 0.016738785430788994,
        "p25": 0.016076551284641027,
        "p50": 0.016747331246733665,
        "p75": 0.01742139644920826,
        "std": 0.0009769928874447942,
        "evaluated_voxels": 714,
    },
    "water": {
        "min": 0.015405772253870964,
        "max": 0.037061210721731186,
        "mean": 0.020344505086541176,
        "p25": 0.019661981612443924,
        "p50": 0.02034671977162361,
        "p75": 0.02102324739098549,
        "std": 0.0010299131972715259,
        "evaluated_voxels": 71310,
    },
}
