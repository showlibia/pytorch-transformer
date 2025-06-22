from pathlib import Path

def get_config():
    return {
        "batch_size": 12,
        "num_epochs": 20,
        "lr": 1e-4,
        "seq_len": 660,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "zh",
        "model_folder": "weigths",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": f"tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weigths_file_path(config, epoch:str):
    model_folder = config["model_folder"]
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)
