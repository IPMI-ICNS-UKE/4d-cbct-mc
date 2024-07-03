from scripts.run_mc_simulations import run
import pathlib as Path
import yaml
image_filepath = Path("media/laura/2TB/4dcbct_sim/306_R2020033/4dct/4DCT_2.mha")
output_folder = Path("/media/laura/2TB/4dcbct_sim/306_R2020033/4d_recon")
correspondence_model = Path("/media/laura/2TB/4dcbct_sim/306_R2020033/correspondence_model_c061c22.pkl")
signal = Path("/media/laura/2TB/4dcbct_sim/306_R2020033/rpm/signal.pkl")
speedups = 10

run(
    image_filepath = image_filepath,
    output_folder = output_folder,
    speedups = speedups,
    reconstruct_4d = True,
    forward_projection = True,
    no_clean = True,
    correspondence_model = correspondence_model,
    respiratory_signal = signal,
    loglevel = "debug",
)