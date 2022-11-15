class ClipModel:
    def __init__(self):
        self.model = None
        self.preprocess = None
        self.device = None

    def get_model_and_preprocess(self):
        if self.model is None:
            self.initialize_model()
        return self.model, self.preprocess, self.device

    def initialize_model(self):
        import torch
        import clip
        # See first example at https://github.com/openai/CLIP#usage
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)


class ResnetModel:
    def __init__(self):
        self.model = None
        self.preprocess = None

    def get_model_and_preprocess(self):
        if self.model is None:
            self.initialize_model()
        return self.model, self.preprocess

    def initialize_model(self):
        import tensorflow as tf
        from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
        from tensorflow.keras.models import Model
        base_model = ResNet50V2(
            include_top=False,
            weights='imagenet',
            input_tensor=None,
            input_shape=None,
            pooling='avg'
        )
        input_layer = tf.keras.Input(shape=(None, None, 3))
        preprocessed_input = preprocess_input(input_layer)
        prediction_layer = base_model(preprocessed_input)
        self.model = Model(inputs=input_layer, outputs=prediction_layer)
        self.preprocess = preprocess_input
