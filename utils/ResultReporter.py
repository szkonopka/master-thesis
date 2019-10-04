class Metrics:
    def __init__(self):
        self.epochs = None
        self.accuracy = []
        self.loss = []


class ModelResult:
    def __init__(self, metrics):
        self.metrics = metrics

class ModelResultReporter:
    def __init__(self, model_name, output_path):
        self.model_name = model_name
        self.output_path

    def __call__(self, model_result):
        self.model_result = model_result

