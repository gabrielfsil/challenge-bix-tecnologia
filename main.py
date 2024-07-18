from src.data.load_data import load_data
from src.data.preprocess import preprocess
from src.models.train_model import train_model
from src.models.evaluate_model import evaluate_model

def main():
    train, val = load_data()
    X_train, X_test, y_train, y_test, X_val, y_val = preprocess(train, val)
    model = train_model(X_train, X_test, y_train, y_test)
    evaluate_model(model, X_val, y_val)

main()