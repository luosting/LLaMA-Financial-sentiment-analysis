# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import sys
sys.path.insert(0, 'llama')
from generation import Llama, Dialog
from typing import List
import tqdm



class BaseModel:
    def __init__(
            self,
            ckpt_dir='dockerdata/llama2_7b_base',
            tokenizer_path='dockerdata/llama2_7b_base/tokenizer.model',
            temperature: float = 0.6,
            top_p: float = 0.9,
            max_seq_len: int = 128,
            max_gen_len: int = 64,
            max_batch_size: int = 4,
    ):
        self.generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        )
        self.temperature = temperature
        self.top_p = top_p
        self.max_gen_len = max_gen_len

    def generate(self, prompt):
        prompts: List[str] = [prompt]
        results = self.generator.text_completion(
            prompts,
            max_gen_len=self.max_gen_len,
            temperature=self.temperature,
            top_p=self.top_p,
            logprobs=True, # mod
        )
        return np.argmax(results.cpu().numpy()) # mod


def main(
    ckpt_dir='dockerdata/llama2_7b_base',
    tokenizer_path='dockerdata/llama2_7b_base/tokenizer.model',
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 128.
        max_gen_len (int, optional): The maximum length of generated sequences. Defaults to 64.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 4.
    """ 
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    prompts: List[str] = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
    ]
    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")

def validation_sft():
    """
    Calculate the validation accuracy.
    """
    # read the prompt
    with open('prompt.txt') as fp:
        prompt = fp.read().strip()
    # read the validation data
    choices = {'positive': 0, 'negative': 1, 'neutral': 2}
    B_INST, E_INST = "[INST]", "[/INST]"
    test_lines = []
    with open('data/test_data.txt') as fp:
        lines = fp.read().strip().split('\n')
        for l in lines:
            label, text = l.split('\t\t')
            label = choices[label]
            question = prompt.format(news_title=text)
            test_lines.append((f'{question}', label))
    # init model
    model = BaseModel(
        #ckpt_dir='/dockerdata/rst_checkpoints/2023_12_13_17_52_19/0.0B',
        ckpt_dir='/dockerdata/rst_checkpoints/2023_12_14_12_07_55/0.01B',
        temperature=0, max_gen_len=1, max_seq_len=1024)
    # perform test
    pbar = tqdm.tqdm(test_lines)
    n_correct = 0
    n_total = 0

    for text, label in pbar:
        ans = model.generate(text)
        n_total += 1
        if ans == label: n_correct += 1
        pbar.set_description(f'Accuracy={n_correct/float(n_total):.3f}')


def validation():
    """
    Calculate the validation accuracy.
    """
    # read the prompt
    with open('prompt.txt') as fp:
        prompt = fp.read().strip()
    # read the validation data
    choices = {'positive': 'A', 'negative': 'B', 'neutral': 'C'}
    test_lines = []
    with open('data/test_data.txt') as fp:
        lines = fp.read().strip().split('\n')
        for l in lines:
            label, text = l.split('\t\t')
            label = choices[label]
            test_lines.append((prompt.format(news_title=text), label))
    # init model
    model = BaseModel(
        #ckpt_dir='/dockerdata/rst_checkpoints/2023_12_13_17_52_19/0.0B',
        ckpt_dir='/dockerdata/rst_checkpoints/2023_12_13_22_39_51/0.0B',
        temperature=0, max_gen_len=1, max_seq_len=1024)
    # perform test
    pbar = tqdm.tqdm(test_lines)
    n_correct = 0
    n_total = 0
    for text, label in pbar:
        ans = model.generate(text)
        n_total += 1
        if ans == label: n_correct += 1
        pbar.set_description(f'Accuracy={n_correct/float(n_total):.3f}')


if __name__ == "__main__":
    validation_sft()
