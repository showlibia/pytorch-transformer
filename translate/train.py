from json import decoder, encoder
from os import write
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm
import warnings

from translate.config import get_config, get_weights_file_path
from translate.dataset import BilingualDataset, causal_mask
from translate.model import build_model
from torch.utils.tensorboard import SummaryWriter

def greedy_decode(model, src, src_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(src, src_mask)
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(src).to(device) 
    while True:
        if decoder_input.size(1) == max_len:
            break

        # Build mask for tgt
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(src).to(device)

        # calculate output
        out = model.decode(encoder_output, src_mask, decoder_input, decoder_mask)
        
        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1).fill_(next_word.item()).type_as(src).to(device),
            ],
            dim=1,
        )

        if next_word == eos_idx:
            break

        return decoder_input.squeeze(0)

def run_validation(model, val_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_state, writer, num_examples=2):
    model.eval()
    count = 0

    with torch.no_grad():
        for batch in val_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)  # (B, Seq_len)
            encoder_mask = batch['encoder_mask'].to(device)  # (B, 1, 1, Seq_len)

            # check the batch size is 1
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(
                model,
                encoder_input,
                encoder_mask,
                tokenizer_src,
                tokenizer_tgt,
                max_len,
                device
            )

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy()) # type: ignore

            print_msg("-" * 100)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg("-" * 100)
                break


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(vocab={}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace() # type: ignore
        trainer = WordLevelTrainer( 
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2  # type: ignore
        )  
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset("Helsinki-NLP/un_ga", f'{config["lang_src"]}_to_{config["lang_tgt"]}', split='train')
    # Only keep 1/2 of the dataset to reduce size
    ds_raw = ds_raw.select(range(0, len(ds_raw), 2))

    # Build tokenizers
    tokenizer_src = get_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt = get_tokenizer(config, ds_raw, config["lang_tgt"])

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(
        train_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config['lang_src'],
        config['lang_tgt'],
        config['seq_len']
    )
    val_ds = BilingualDataset(
        val_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config['lang_src'],
        config['lang_tgt'],
        config['seq_len']
    )

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of source sentence: {max_len_src}")
    print(f"Max length of target sentence: {max_len_tgt}")

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_model(vocab_src_len, vocab_tgt_len, config["seq_len"], config["seq_len"], d_model=config["d_model"])
    return model

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, len(tokenizer_src.get_vocab()), len(tokenizer_tgt.get_vocab())).to(device)

    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9 )

    initial_epoch = 0
    global_step = 0
    if config['preload'] is not None:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Loading model from {model_filename}")
        state = torch.load(model_filename)
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id("[PAD]"), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{config["num_epochs"]}', unit='batch')
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # (B, Seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, Seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, Seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, Seq_len, Seq_len)

            # Run the tensors through the encoder, decoder and projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, Seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, Seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, Seq_len)
            
            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1)) 
            batch_iterator.set_postfix({"loss": f"{loss.item(): 6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
        
            run_validation(
                model,
                val_dataloader,
                tokenizer_src,
                tokenizer_tgt,
                config['seq_len'],
                device,
                print_msg=lambda msg: batch_iterator.write(msg),
                global_state=global_step,
                writer=writer
            )

        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
