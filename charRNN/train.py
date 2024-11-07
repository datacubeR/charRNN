import torch
import numpy as np
from .model_utils import (
    Tokenizer,
    import_text,
    create_vocabulary,
    QuijoteSeqDataset,
    CharRNN,
    fit_model,
)
from sklearn.model_selection import train_test_split
import argparse

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

text = import_text("el_quijote.txt")
vocabulary = create_vocabulary(text)
tokenizer = Tokenizer(vocabulary)
encoded_text = tokenizer.text_to_seq(text)

text_train, text_val = train_test_split(
    encoded_text, test_size=0.2, random_state=RANDOM_SEED, shuffle=False
)


model = CharRNN(tokenizer.n_vocabulary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train CharRNN model for Text Generation"
    )
    parser.add_argument(
        "-bst",
        "--batch_size_train",
        type=int,
        default=512,
        help="Batch size for training",
    )
    parser.add_argument(
        "-bsv",
        "--batch_size_val",
        type=int,
        default=2048,
        help="Batch size for validation",
    )
    parser.add_argument("-lr", "--learning_rate", type=float, default=3e-4)
    parser.add_argument("-e", "--epochs", type=int, required=True)
    parser.add_argument("-w", "--window_size", type=int, required=True)
    args = parser.parse_args()

    dataset = dict(
        train=QuijoteSeqDataset(text_train, window_size=args.window_size),
        val=QuijoteSeqDataset(text_val, window_size=args.window_size),
    )

    training_params = dict(
        batch_size_train=args.batch_size_train,
        batch_size_val=args.batch_size_val,
        lr=args.learning_rate,
        epochs=args.epochs,
    )
    model, loss = fit_model(model, dataset, training_params, device)
