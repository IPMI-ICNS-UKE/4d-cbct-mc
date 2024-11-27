from pathlib import Path

CONFIG = {
    "clemens": {
        "root_folder": Path("/home/crohling/amalthea/data"),
        "device": "cuda:0",
    },
    "frederic": {
        "root_folder": Path("/mnt/gpu_server/crohling/data"),
        "device": "cuda:1",
    },
}


def get_user_config():
    for user, config in CONFIG.items():
        if config["root_folder"].exists():
            return user, config
