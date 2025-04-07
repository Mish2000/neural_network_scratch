import os
import cupy as np
from data_utils.text_preprocessing import (
    simple_tokenize,
    build_vocabulary,
    batch_texts_to_sequences, text_to_sequence
)

from model.model import Model
from layers.layer_dense import Layer_Dense
from layers.layer_dropout import Layer_Dropout
from layers.layer_embedding import Layer_Embedding
from activations.activation_relu import Activation_ReLU
from activations.activation_softmax import Activation_Softmax
from losses.loss_categorical_crossentropy import Loss_CategoricalCrossentropy
from optimizers.optimizer_adam import Optimizer_Adam
from accuracy.accuracy_categorical import Accuracy_Categorical

def load_text_dataset():
    texts = [
       "I love this product",
       "This is terrible",
       "Amazing experience",
       "Worst purchase ever",
       "Very happy with the results",
       "I do not like it at all"
    ]
    labels = [1, 0, 1, 0, 1, 0]

    return texts, labels

def main():
    texts, labels = load_text_dataset()

    vocab = build_vocabulary(texts, min_freq=1)
    vocab_size = len(vocab)
    print("Vocab size:", vocab_size)

    max_len = 5
    X = batch_texts_to_sequences(texts, vocab, max_len)
    y = np.array(labels, dtype=np.int32)

    model = Model()
    model.add(Layer_Embedding(vocab_size=vocab_size, embed_dim=8))
    model.add(Layer_Dropout(rate=0.2))
    model.add(Layer_Dense(n_inputs=max_len*8, n_neurons=16))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(16, 2))
    model.add(Activation_Softmax())

    model.set(
        loss=Loss_CategoricalCrossentropy(),
        optimizer=Optimizer_Adam(learning_rate=0.001),
        accuracy=Accuracy_Categorical()
    )

    model.finalize()

    X_emb_input = X

    model.train(X_emb_input, y, epochs=3, batch_size=2, print_every=1)

    new_text = "I really love this!"
    seq_new = text_to_sequence(new_text, vocab, max_len)
    seq_new = np.expand_dims(seq_new, axis=0)
    preds = model.predict(seq_new)
    print("Prediction probabilities:", preds)

    saves_dir = os.path.join("..", "saves")
    if not os.path.exists(saves_dir):
        os.makedirs(saves_dir)
    model_save_path = os.path.join(saves_dir, "text_model_parameters.pkl")
    model.save_parameters(model_save_path)


if __name__ == "__main__":
    main()
