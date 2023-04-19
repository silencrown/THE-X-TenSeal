import sys
from pathlib import Path
import argparse
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np

from thex import cxt_man
from thex import logger
from thex.xnn.softmax import (
    SoftmaxApprox,
    EncSoftmax
)
from thex.convert.softmax_approx import (
    SoftmaxApproxTrainer
)

def train_softmax(args):
    softmax_model = SoftmaxApprox(hidden_size=args.softmax_hidden_size)
    softmax_trainer = SoftmaxApproxTrainer(
                                softmax_model, 
                                num_samples=args.softmax_num_samples, 
                                input_size=args.softmax_input_size,
                                batch_size=args.softmax_batch_size, 
                                lr=args.softmax_lr, 
                                num_epochs=args.softmax_num_epochs)

    args = parser.parse_args()
    softmax_trainer.train()
    softmax_trainer.save(file_path=args.softmax_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Toy Bert Test")

    """ softmax approx args """
    parser.add_argument("--is_softmax", type=bool, default=False, help="Whether to train the SoftmaxApprox model")
    parser.add_argument("--softmax_model_path", type=str, default="../cache/softmax_approx.pt", help="Path to save the SoftmaxApprox model")
    parser.add_argument("--softmax_hidden_size", type=int, default=16, help="Hidden size for the SoftmaxApprox model")
    parser.add_argument("--softmax_num_samples", type=int, default=int(1e6), help="Number of samples for training")
    parser.add_argument("--softmax_input_size", type=int, default=128, help="Input size for the SoftmaxApprox model")
    parser.add_argument("--softmax_batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--softmax_lr", type=float, default=0.00001, help="Learning rate for training")
    parser.add_argument("--softmax_num_epochs", type=int, default=100, help="Number of epochs for training")
    
    args = parser.parse_args()

    if args.is_softmax:
        train_softmax(args)
