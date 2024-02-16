from typing import List
import os
import time
import torch
import random
from dataclasses import dataclass
from torch.functional import Tensor
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
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
	
	def __getitem__(self, idx):
		if self.max_word_length == -1:
			word = self.words[idx]
			ix = self.encode(word)
			x = torch.zeros(len(ix) + 1, dtype=torch.long)
			y = torch.zeros(len(ix) + 1, dtype=torch.long)
			x[1:] = ix
			y[:-1] = ix
			y[-1] = 0
			return x, y

		word = self.words[idx]
		ix = self.encode(word)
		x = torch.zeros(self.max_word_length + 1, dtype=torch.long)
		y = torch.zeros(self.max_word_length + 1, dtype=torch.long)
		x[1:1+len(ix)] = ix
		y[:len(ix)] = ix
		y[len(ix)+1:] = 0 # index -1 will mask the loss at the inactive locations

		return x, y

def create_dataset(input_file, use_window=True) -> tuple[CharDataset, CharDataset]:
	with open(input_file, 'r') as f:
		words = f.read().splitlines()
	names = [w.strip() for w in words]

	max_len = max(len(v) for v in names)
	min_len = min(len(v) for v in names)
	chars = sorted(list(set("".join(names))))
	chars_count = len(chars)
	test_set_size = min(1000, int(len(names) * 0.1))

	names = sorted(names, key=lambda _: random.random())
	train_dset = CharDataset(names[:-test_set_size], chars, max_len if use_window else -1)
	test_dset = CharDataset(names[-test_set_size:], chars, max_len if use_window else -1)

	print("\n", "*"*30, "DATASET INFO", "*"*30)
	print("names: ", names[:5])
	print("number of names: ", len(names))
	print("vocab_size: ", chars_count)
	print("context length: ", max_len)
	print("(list of chars, vocab_size): ", ("".join(chars), chars_count))
	print("(max word length, min word length): ", (max_len, min_len))
	print("(train size, test size): ", (len(names) - test_set_size, test_set_size))
	print("*" * 75, "\n")

	return train_dset, test_dset

# a dataloader that can be iterated infinitely
class InfiniteDataLoader:
    def __init__(self, dataset, **kwargs):
        train_sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=int(1e10))
        self.train_loader = DataLoader(dataset, sampler=train_sampler, **kwargs)
        self.data_iter = iter(self.train_loader)

    def next(self):
        try:
            batch = next(self.data_iter)
        except StopIteration: # this will technically only happen after 1e10 samples... (i.e. basically never)
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        return batch

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
		return logits, loss 

# -----------------------------------------------------------------------------

@torch.no_grad()
def generate(model:nn.Module, idx, max_new_tokens, do_sample=False, tempreture=1.0):
	block_size = model.get_block_size()
	for _ in range(max_new_tokens):
		idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
		logits, _ = model(idx_cond)
		# pluck the logits at the final step and scale by desired temperature
		logits = logits[:, -1, :] / tempreture

		if top_k is not None:
			k, _ = torch.top_k(logits, top_k=top_k)
			logits[logits < k[:, [-1]]] = -float('Inf')

		if do_sample:
			idx_next = torch.multinomial(probs, num_samples=1)
		else:
			_, idx_next = torch.topk(logits, k=1, dim=-1)

		probs = F.softmax(logits, dim=1)
		idx = torch.cat((idx, idx_next), dim=1)
	return idx
	

def print_samples(num=10):
	idx = torch.zeros(num, 1, dtype=torch.long).to("cpu")
	steps = train_dset.get_output_length() - 1 # -1 because we already start with <START> token (index 0)

	samples = generate(model, idx, steps)
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
	sets = { "train_samp": train_samp, "test_samp": test_samp, "new_samp":new_samp}

	print("*"*10)
	for desc, lst in sets.items():
		print("*"*3, desc, "*"*3)
		for word in lst:
			print(word)
	print("*"*10)
	

@torch.inference_mode()
def evaluate(model, dataset, batch_size=None, max_batches=None):
	model.eval()
	loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
	losses = []
	for i, batch in enumerate(loader):
		X, Y = batch
		_, loss = model(X, Y)
		losses.append(loss.item())
		if max_batches is not None and i > max_batches:
			break
	avg_loss = sum(losses) / len(losses)
	model.train()
	return avg_loss
		

if __name__ == "__main__":
	top_k = None
	input_file = "./assets/names.txt"
	work_dir = "./assets"
	batch_size = 32
	num_workers = 2
	max_steps = 1000
	seed = 2147483647
	block_size = 3
	learning_rate = 0.1
	weight_decay = 0.01

	torch.manual_seed(seed)
	writer = SummaryWriter(log_dir=os.path.join(work_dir, "logs"))

	os.makedirs(work_dir, exist_ok=True)
	writer = SummaryWriter(log_dir=work_dir)

	train_dset, test_dset = create_dataset(input_file)
	vocab_size = train_dset.get_vocab_size()
	block_size = min(block_size, train_dset.get_output_length()) # context length

	conf = ModelConfig(
			vocab_size=vocab_size,
			block_size = block_size,
			)

	model = Bigram(conf)

#	print("params: ", model.parameters())
	
	state_path = os.path.join("assets", "model", "model.pt")
#	model.load_state_dict(torch.load(state_path))

	print("Model training...")
	# init optimizer
	optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.99), eps=1e-8)
	# init dataloader
	batch_loader = InfiniteDataLoader(train_dset, batch_size=batch_size, pin_memory=True, num_workers=num_workers)
	# print_samples()

	lre = torch.arange(-3, 0, max_steps)
	lri = 10 ** lre 
	best_loss = None
	losses = []
	for i in range(max_steps):
		start_time = time.time()

		# minibatches
		batch = batch_loader.next()
		X, Y = batch

		# forwardpass
		logits, loss = model(X, Y)


		# # backward pass
		model.zero_grad(set_to_none=True)
		loss.backward()

		# # update
		optimizer.step()

		step_time = time.time() - start_time

		# track stats
		if i % 100 == 0:
			print(f"step {i} | loss {loss.item():.4f} | step time {step_time*1000:.2f}ms")

        # evaluate the model
		if i > 0 and i% 500 == 0:
			train_loss = evaluate(model, train_dset, batch_size=100, max_batches=10)
			test_loss  = evaluate(model, test_dset,  batch_size=100, max_batches=10)
			writer.add_scalar("Loss/train", train_loss, i)
			writer.add_scalar("Loss/test", test_loss, i)
			writer.flush()
			print(f"step {i} | train loss: {train_loss} | test loss: {test_loss}")
			# save the model to disk if it has improved
			if best_loss is None or test_loss < best_loss:
				out_path = os.path.join(work_dir, "model", "model.pt")
				print(f"test loss {test_loss} is the best so far, saving model to {out_path}")
				torch.save(model.state_dict(), out_path)
				best_loss = test_loss

	if i > 0 and i % 1000 == 0:
		print_samples(num=100)
