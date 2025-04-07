import cupy as np

class Layer_Embedding:

    def __init__(self, vocab_size, embed_dim):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embeddings = 0.01 * np.random.randn(vocab_size, embed_dim)

    def forward(self, inputs, training):
        self.inputs = inputs
        batch_size, seq_len = inputs.shape
        self.output = np.zeros((batch_size, seq_len, self.embed_dim), dtype=np.float32)
        for i in range(batch_size):
            for j in range(seq_len):
                token_idx = inputs[i, j]
                self.output[i, j] = self.embeddings[token_idx]

    def backward(self, dvalues):
        self.dembeddings = np.zeros_like(self.embeddings)

        batch_size, seq_len, _ = dvalues.shape
        for i in range(batch_size):
            for j in range(seq_len):
                token_idx = self.inputs[i, j]
                self.dembeddings[token_idx] += dvalues[i, j]

        self.dinputs = None

    def get_parameters(self):
        return (self.embeddings,), None

    def set_parameters(self, weights, biases):
        self.embeddings = weights[0]
