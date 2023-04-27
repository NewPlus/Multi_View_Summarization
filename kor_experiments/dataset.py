import random
from transformers import AutoTokenizer
from datasets import load_dataset
import re

data = load_dataset("csv", data_files={"train": "/root/bart_customize/data/train.csv", "valid": "/root/bart_customize/data/valid.csv", "test": "/root/bart_customize/data/test.csv"})
print(f"{data}")