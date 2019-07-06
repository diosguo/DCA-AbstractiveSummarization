from tensorflow.python.keras.layers import Embedding, CuDNNLSTM, Dropout, Dense, Bidirectional, Flatten
from tensorflow.python.keras.models import Sequential

class DCA_Model(object):
    def __init__(self, arg):
        self.n_agents = arg.agents_num

    def _build_local_encoder(self):
        self._local_encoder = []
        for i in range(self.n_agents):
            e = Embedding(400, 64, input_length=400)
            e = Dropout(0.2)(e)
            b_lstm = Bidirectional(CuDNNLSTM(64, return_sequences=True), merge_mode='concat')
            b_lstm = Dropout(b_lstm)
            self._local_encoder.append(b_lstm)

    def _build_contextual_encoder(self):
        self._contextual_encoder = []
        for i in range(self.n_agents):




