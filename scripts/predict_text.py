import os
import cupy as np
from data_utils.text_preprocessing import build_vocabulary, text_to_sequence
from model.model import Model

def prepare_text(text, vocab, max_len=20):
    seq = text_to_sequence(text, vocab, max_len)
    seq = np.expand_dims(seq, axis=0)
    return seq

def make_prediction(text, vocab, model, max_len=20):
    input_seq = prepare_text(text, vocab, max_len)
    prediction = model.predict(input_seq)
    return prediction

if __name__ == "__main__":
    model_path = os.path.join("..", "saves", "text_model_parameters.pkl")
    model = Model()

    text_sample = "I do not like this product"
    preds = make_prediction(text_sample, vocab, model, max_len=20)
    print("Prediction:", preds)
