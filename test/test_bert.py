import logging
import argparse
import tqdm
import torch
from torch.utils.data import DataLoader
from src.model.bert.model import BERT, BERTLM
from src.model.bert.enc_model import BERTEnc
from src.model.bert.dataset import BERTDataset, WordVocab


def load_bert_model(args, vocab):
    """get origin bert model"""

    print("Building BERT model")
    bert = BERT(len(vocab), hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads)

    print("Loading model from", args.output_path)

    bert.load_state_dict(torch.load(args.output_path))
    logging.info("model loaded from {}".format(args.output_path))
    logging.info("model: {}".format(bert))
    bert.eval()
    return bert

def test_iter(model, data_loader):
    data_iter = tqdm.tqdm(enumerate(data_loader),
                        desc="EP_%s" % ("test"),
                        total=len(data_loader),
                        bar_format="{l_bar}{r_bar}")
    for i, data in data_iter:
        data = {key: value.to('cpu') for key, value in data.items()}
        next_sent_output, mask_lm_output = model.forward(data["bert_input"], data["segment_label"])

def main():
    DATA_PATH = "/home/gaosq/THE-X-TenSEAL/test/data/"

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--train_dataset", default=DATA_PATH+"SST-2/train.tsv", type=str, help="train dataset for train bert")
    parser.add_argument("-t", "--test_dataset", default=DATA_PATH+"SST-2/test.tsv", type=str, help="test set for evaluate train set")
    parser.add_argument("-v", "--vocab_path", default=DATA_PATH+"vocab/vocab.txt", type=str, help="built vocab model path with bert-vocab")
    parser.add_argument("-o", "--output_path", default=DATA_PATH+"output/bert.model", type=str, help="ex)output/bert.model")

    parser.add_argument("-hs", "--hidden", type=int, default=256, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=8, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=8, help="number of attention heads")
    parser.add_argument("-s", "--seq_len", type=int, default=20, help="maximum sequence len")

    parser.add_argument("-b", "--batch_size", type=int, default=64, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=5, help="dataloader worker size")

    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=10, help="printing loss every n iter: setting n")
    parser.add_argument("--corpus_lines", type=int, default=None, help="total number of lines in corpus")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")

    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    args = parser.parse_args()

    # get test dataset
    print("Loading Vocab", args.vocab_path)
    vocab = WordVocab.load_vocab(args.vocab_path)
    print("Vocab Size: ", len(vocab))
    print("Loading Test Dataset", args.test_dataset)
    test_dataset = BERTDataset(args.test_dataset, vocab, seq_len=args.seq_len, on_memory=args.on_memory)
    print("Creating Dataloader")
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    # get origin model
    bert = load_bert_model(args, vocab)
    model = BERTLM(bert, len(vocab))
    test_iter(model, test_data_loader)

    # get encryped model
    enc_bert = load_bert_model(args, vocab)
    enc_model = BERTLM(enc_bert, len(vocab))
    test_iter(enc_model, test_data_loader)

if __name__ == "__main__":
    main()