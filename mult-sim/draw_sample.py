import json
import argparse
from tqdm import tqdm
import math
import itertools
import random
from random import randrange
from pathlib import Path
import csv

random.seed(0)

def normalize(number, min_value, max_value):
    """Normalize a number to the range [-5, 5]."""
    # return (number - min_value) / (max_value - min_value) * 10 - 5
    log_min_val = math.log(min_value)
    log_max_val = math.log(max_value)
    log_value = math.log(number)
    return (log_value - log_min_val) / (log_max_val - log_min_val)

def denormalize(normalized_number, min_value, max_value):
    """Denormalize a number from the range [-5, 5] back to its original scale."""
    # return (normalized_number + 5) / 10 * (max_value - min_value) + min_value
    log_min_val = math.log(min_value)
    log_max_val = math.log(max_value)
    log_value = normalized_number * (log_max_val - log_min_val) + log_min_val
    return math.exp(log_value)

def all_n_digit(num_digit):
    return list(range(int(math.pow(10, num_digit - 1)), int(math.pow(10, num_digit))))


def random_n_digit(num_digit):
    return randrange(int(math.pow(10, num_digit - 1)), int(math.pow(10, num_digit)))


def cartesian(a_num_digit, b_num_digit):
    a_numbers, b_numbers = all_n_digit(a_num_digit), all_n_digit(b_num_digit)
    inputs = [e for e in itertools.product(a_numbers, b_numbers)]
    return inputs

def sample(a_num_digit, b_num_digit, max_sequence):
    inputs = set()
    while len(inputs) < max_sequence:
        a, b = random_n_digit(a_num_digit), random_n_digit(b_num_digit)
        if (a, b) not in inputs:
            inputs.add((a, b))
    return list(inputs)

def construct_dataset(num_digit, max_sequence):
    digits = list(range(1, num_digit + 1))
    datasets = {}
    for a_num_digit in digits:
        for b_num_digit in tqdm(digits[:a_num_digit]):
            name = f'{a_num_digit}_by_{b_num_digit}'
            num_combination = math.pow(10, a_num_digit + b_num_digit)
            if num_combination < max_sequence:
                inputs = cartesian(a_num_digit, b_num_digit)
                random.shuffle(inputs)
            else:
                inputs = sample(a_num_digit, b_num_digit, max_sequence)
            datasets[name] = inputs
    return datasets

num_digit = 5
max_sequence = 125000
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Dynamically calculate min and max values for normalization based on num_digit
min_value = 1  # Minimum multiplicand value
max_value = int(math.pow(10, num_digit)) - 1  # Maximum multiplicand value based on num_digit
product_min_value = 1  # Minimum product value
product_max_value = (max_value ** 2)  # Maximum possible product value

datasets = construct_dataset(num_digit, max_sequence)


def reformat_input(inputs):
    a, b = inputs
    # Normalize multiplicands
    # a_norm = normalize(a, min_value, max_value)
    # b_norm = normalize(b, min_value, max_value)
    # Calculate and normalize product
    product = a * b
    product_norm = normalize(product, product_min_value, product_max_value)
    str_input = f'{a} times {b} is {product_norm}'
    return str_input


# hold out val and test
train_datasets = {}
val_datasets = {}
test_datasets = {}
for name, data in datasets.items():
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_datasets[name] = data[:train_end]
    val_datasets[name] = data[train_end:val_end]
    test_datasets[name] = data[val_end:]
    
all_train_data = []
for name, data in train_datasets.items():
    for d in data:
        all_train_data.append(reformat_input(d))

with open('multi_train', "w") as f:
    for s in all_train_data:
        # write each item on a new line
        f.write("{}\n".format(s))
        
for name, data in val_datasets.items():
    val_datasets[name] = [reformat_input(d) for d in data]

# write each item on a new line, each dataset in a separate file
for name, data in val_datasets.items():
    with open(f'multi_val_{name}', "w") as f:
        for s in data:
            f.write("{}\n".format(s))
            
for name, data in test_datasets.items():
    test_datasets[name] = [reformat_input(d) for d in data]
    
# write each item on a new line, each dataset in a separate file
for name, data in test_datasets.items():
    with open(f'multi_test_{name}', "w") as f:
        for s in data:
            f.write("{}\n".format(s))