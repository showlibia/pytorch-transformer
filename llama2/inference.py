from ctypes.macholib import dyld
from re import L
import token
import torch
from typing import Optional
import time

from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from torch._higher_order_ops import out_dtype
from tqdm import tqdm

from model import ModelArgs, Transformer

class LlaMA:
    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, args: ModelArgs) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

    @staticmethod
    def build(checkpoints_path: str, tokenizer_path: str, load_model: bool, max_seq_len: int, max_batch_size: int, device: str):
        start_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoints_path).glob("*.pth"))
            assert len(checkpoints) > 0, "No checkpoints found in the specified path."
            ckpt_path = checkpoints[0]
            print(f'Loading checkpoint from {ckpt_path}')
            checkpoint = torch.load(ckpt_path)
            print(f'Checkpoint loaded in {time.time() - start_time:.2f} seconds')
            start_time = time.time()
        with open(Path(checkpoints_path) / 'params.json', 'r') as f:
            params = json.load(f)

        args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
        )

        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        args.vocab_size = tokenizer.vocab_size()

        if device == 'cuda':
            torch.set_default_dtype(torch.float16)
        else:
            torch.set_default_dtype(torch.bfloat16)

        model = Transformer(args).to(device)

        if load_model:
            del checkpoint['rope.freqs']
            model.load_state_dict(checkpoint, strict=True)
            print(f"Loaded state dict in {time.time() - start_time:.2f} seconds")

        return LlaMA(model, tokenizer, args)
    
    def text_completion(self, prompts: list[str], temperature: float = 0.6, top_p: float = 0.9, max_gen_len: Optional[int] = None):
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1
        # Conver each prompt to tokens
        prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]
        # Make sure the batch size is not too large
        batch_size = len(prompt_tokens)
        assert batch_size <= self.args.max_batch_size, f"Batch size {batch_size} exceeds max batch size {self.args.max_batch_size}"

        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        # Make sure the prompt length is not longer than the max sequence length
        assert max_prompt_len <= self.args.max_seq_len, f"Max prompt length {max_prompt_len} exceeds max sequence length {self.args.max_seq_len}"
        total_len = min(self.args.max_seq_len, max_prompt_len + max_gen_len)

        # Create the list that will contain the generated tokens
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)
        for k, t in enumerate(prompt_tokens):
            tokens[k, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)

        eos_reached = torch.tensor([False] * batch_size, device=device)
        prompt_tokens_mask = tokens != pad_id
        cur_iterator = tqdm(range(1, total_len), desc="Generating tokens")
        for cur_pos in cur_iterator:
            with torch.no_grad():
                logits = self.model.forward(tokens[:, cur_pos - 1: cur_pos], start_pos=cur_pos)
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)
            else:
                # Greedy decoding
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # Only replace the token if it is a padding token
            next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            # EOS is reached only if we found an EOS token for a padding postion
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id())
            if all(eos_reached):
                break

        out_tokens = []
        out_text = []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            # Cut to the EOS 
            if self.tokenizer.eos_id() in current_prompt_tokens:
                current_prompt_tokens = current_prompt_tokens[:current_prompt_tokens.index(self.tokenizer.eos_id())]
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))
        return out_tokens, out_text
    
    def _sample_top_p(self, probs: torch.Tensor, top_p: float):
        probs_sorted, original_indices = torch.sort(probs, dim=-1,descending=True)
        probs_sum = torch.cumsum(probs_sorted, dim=-1)
        # Find the index where the cumulative probability just exceeds top_p
        mask = probs_sum - probs_sorted > top_p
        probs_sorted[mask] = 0.0
        # Normalize the probabilities
        probs_sorted.div_(probs_sorted.sum(dim=-1, keepdim=True))
        # Sample a token(index) from the top-p distribution
        next_token = torch.multinomial(probs_sorted, num_samples=1)
        # Get the token position in the original order
        next_token = torch.gather(original_indices, -1, next_token)
        return next_token

if __name__ == "__main__":
    torch.manual_seed(0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    prompts = [
        "Simply put, the theory of relativity states that ",
        "If Google was an Italian company founded in Milan, it would",
        # Few shot promt
        """Translate English to French:
        
        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
        # Zero shot prompt
        """Tell me if the following person is actually Doraemon disguised as human:
        Name: Umar Jamil
        Decision: 
        """,
    ]

    model = LlaMA.build(
        checkpoints_path='llama-2-7b/',
        tokenizer_path='tokenizer.model',
        load_model=True,
        max_seq_len=1024,
        max_batch_size=len(prompts),
        device=device
    )

    out_tokens, out_text = model.text_completion(prompts, max_gen_len=64)
    assert len(out_tokens) == len(prompts)
    for i in range(len(out_text)):
        print(f'{out_text[i]}')
        print('-'*80)