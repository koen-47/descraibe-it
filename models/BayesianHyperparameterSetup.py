class BayesianHyperparameterSetup:
    def __init__(self, model, n_trials):
        self.model = model
        self.n_trials = n_trials

    def get_lstm_hyperparameters(self):
        n_lstm_units = {"name": "n_lstm_units", "min": 16, "max": 512, "step": 32}
        n_fc_layers = {"name": "n_fc_layers", "min": 1, "max": 5, "step": 1}
        n_fc_units = {"name": "n_fc_units", "min": 16, "max": 2048, "step": 64}
        dropout_p = {"name": "dropout_p", "min": 0., "max": 0.9, "step": 0.1}
        return n_lstm_units, n_fc_layers, n_fc_units, dropout_p
