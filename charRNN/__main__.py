import torch
from .model_utils import CharRNN, Tokenizer, import_text, create_vocabulary
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"

text = import_text("el_quijote.txt")
vocabulary = create_vocabulary(text)
tokenizer = Tokenizer(vocabulary)


def predict(model, encoded_text):
    model.eval()
    with torch.no_grad():
        X = torch.tensor(encoded_text).unsqueeze(0).to(device)
        pred = model(X)
    return pred


def generate_text(model, initial_text, chars_to_generate):
    for _ in range(chars_to_generate):
        X_encoded = tokenizer.text_to_seq(initial_text[-100:])
        y_pred = predict(model, X_encoded)
        y_pred = torch.argmax(y_pred, axis=1).item()
        initial_text += tokenizer.seq_to_text([y_pred])

    return initial_text


def generate_probabilistic_text(
    model, initial_text, chars_to_generate, temp=1
):
    for i in range(chars_to_generate):
        X_new_encoded = tokenizer.text_to_seq(initial_text[-100:])
        y_pred = predict(model, X_new_encoded)
        y_pred = y_pred.view(-1).div(temp).exp()
        top_i = torch.multinomial(y_pred, 1).item()
        predicted_char = tokenizer.seq_to_text([top_i])
        initial_text += predicted_char
    return initial_text


model = CharRNN(tokenizer.n_vocabulary)
model.to(device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate text with CharRNN model"
    )
    parser.add_argument("-cp", "--checkpoint", type=str, required=True)
    parser.add_argument("-p", "--prob", action="store_true")
    parser.add_argument("-t", "--temp", type=float, default=1)
    parser.add_argument("-c", "--chars", type=int, default=100)

    args = parser.parse_args()
    initial_text = input("Enter initial text: ")

    model.load_state_dict(
        torch.load(f"checkpoints/{args.checkpoint}.pth", weights_only=True)
    )
    if args.prob:
        print(
            f"Generating {args.chars} characters probabilitically with temperature {args.temp}..."
        )
        output = generate_probabilistic_text(
            model, initial_text, args.chars, args.temp
        )
    else:
        print(f"Generating {args.chars} characters...")
        output = generate_text(model, initial_text, args.chars)

    print("===========================================================")
    print(output)
