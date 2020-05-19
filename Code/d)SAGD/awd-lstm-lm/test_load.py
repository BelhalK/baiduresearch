from main import model_load
import sys

if __name__ == "__main__":
    model_path = sys.argv[1]
    model, criterion, optimizer = model_load(model_path)

    named_params = model.named_parameters()
    for name, param in named_params:
        print(name)