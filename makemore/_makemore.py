from typing import List
import os
import torch
import random
from dataclasses import dataclass
from torch.functional import Tensor
from torch.nn import functional as F
import torch.nn as nn
from torch.utils.data import Dataset

@dataclass
class ModelConfig:
    block_size: int = 0 # length of the input sequences of integers
    vocab_size: int =  0 # the input integers are in range [0 .. vocab_size -1]

    n_layer: int = 4 # number of layers
    n_embd: int = 64 # number of feature channels in the model
    n_embd2: int = 64 # number of feature channels elsewhere in the model

class CharDataset(Dataset):
	def __init__(self, words:List[str], chars, max_word_length):
		self.words = words
		self.chars = chars
		self.max_word_length = max_word_length
		self.atoi = {ch:i+1 for i,ch in enumerate(chars)} # {char: integer}
		self.atoi["."] = 0
		self.itoa = {i:s for s,i in self.atoi.items()} # inverse mapping {integer: char}
	
	def __len__(self):
		return len(self.words)
	
	def contains(self, word):
		return word in self.words
	
	def get_vocab_size(self):
		return len(self.chars) + 1 # all the possible characters and special 0 token

	def get_output_length(self):
		return self.max_word_length + 1 # <START> token followed by words

	def encode(self, word: str) -> Tensor:
		ix = torch.tensor([self.atoi[w] for w in word], dtype=torch.long)
		return ix

	def decode(self, ix: List[int]) -> str:
		return ''.join(self.itoa[i] for i in ix)
	
	def __getitem__(self):
		#TODO
		return


# -----------------------------------------------------------------------------
# Bigram Language Model

class Bigram(nn.Module):
	
	def __init__(self, config:ModelConfig):
		self.block_size = 1 # since this is a 2-gram, onlt prev character is neccessary 
		super().__init__()
		n = config.vocab_size
		self.logits = nn.Parameter(torch.zeros((n, n)))

	def get_block_size(self):
		return self.block_size 

	# TODO: I understand nothing here!!!
	def forward(self, idx, targets=None):

		 # forward pass
		logits = self.logits[idx] # x @ C

		loss = None
		if targets is not None:
			loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
			print("loss shape: ", loss.shape)

		return logits, loss 

# -----------------------------------------------------------------------------
def create_dataset(input_file):
	with open(input_file, 'r') as f:
		words = f.read().splitlines()
	names = [w.strip() for w in words]

	min_chars = 1,
	max_chars = max(len(v) for v in names)
	chars = sorted(list(set("".join(names))))
	chars_count = len(chars)
	print("names: ", names[:5])
	print("number of names: ", len(names))
	print("(list of chars, count): ", ("".join(chars), chars_count))
	print("(max word length, min word length): ", (max_chars, min_chars))

	# TODO: partition
	test_set_size = min(1000, int(len(names) * 0.1))

	names = sorted(names, key=lambda _: random.random())
	train_dset = CharDataset(names[:-test_set_size], chars, max_chars)
	test_dset = CharDataset(names[-test_set_size:], chars, max_chars)

	return train_dset, test_dset

@torch.no_grad()
def generate(model:nn.Module, idx, max_new_tokens, tempreture=-1.0):
	block_size = model.get_block_size()
	for _ in range(max_new_tokens):
		idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
		logits, _ = model(idx_cond)
		# pluck the logits at the final step and scale by desired temperature
		logits = logits[:, -1, :] / tempreture


		probs = F.softmax(logits, dim=1)
		idx_next = torch.multinomial(probs, num_samples=1)
		idx = torch.cat((idx, idx_next), dim=1)
	
	samples = idx
	train_samp, test_samp, new_samp = [], [], []
	for i in range(samples.size(0)):
		row = samples[i,:].tolist()
		word = train_dset.decode(row)
		
		if train_dset.contains(word):
			train_samp.append(word)
		elif test_dset.contains(word):
			test_samp.append(word)
		else:
			new_samp.append(word)
	return { "train_samp": train_samp, "test_samp": test_samp, "new_samp":new_samp}

def print_samples(num=10):
	idx = torch.zeros(num, 1, dtype=torch.long).to("cpu")
	steps = train_dset.get_output_length() - 1 # -1 because we already start with <START> token (index 0)
	samples = generate(model, idx, steps)

	print("*"*10)
	for desc, lst in samples.items():
		print("*"*3, desc, "*"*3)
		for word in lst:
			print(word)
	print("*"*10)
	


if __name__ == "__main__":
	top_k = None
	input_file = "./names.txt"
	batch_size = 32
	num_workers = 2
	learning_rate = 0.1
	weight_decay = 0.01

	torch.manual_seed(2147483647)

	train_dset, test_dset = create_dataset(input_file)
	vocab_size = train_dset.get_vocab_size()
	block_size = train_dset.get_output_length() # context length

	print(f"dataset determined that:\n{vocab_size=}")
	conf = ModelConfig(
			vocab_size=vocab_size,
			block_size = block_size,
			)
	model = Bigram(conf)
	block_size = model.block_size

	print(f"{block_size=}")

	print("params: ", model.parameters())

	
	state_path = os.path.join("assets", "model.pt")
#	model.load_state_dict(torch.load(state_path))

	print("Model training...")
	    # init optimizer
	optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.99), eps=1e-8)

	    # init dataloader
	batch_loader = InfiniteDataLoader(train_dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers)

	print_samples()


