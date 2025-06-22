from datasets import load_dataset


def get_ds(config):
    ds_raw = load_dataset("Helsinki-NLP/un_ga", f'{config["lang_src"]}_to_{config["lang_tgt"]}', split='train')
