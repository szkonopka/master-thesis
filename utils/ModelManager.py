import tensorflow as tf

class ModelManager:
    def load(self, is_tensorflow_model, model_path):
        if is_tensorflow_model:
            return self.load_tensorflow_model(model_path)

    def save(self, is_tensorflow_model, model, model_path):
        if is_tensorflow_model:
            self.save_tensorflow_model(model, model_path)

    def load_tensorflow_model(self, model_path):
        return tf.keras.models.load_model(model_path)

    def save_tensorflow_model(self, model, model_path):
        model.save(model_path)

