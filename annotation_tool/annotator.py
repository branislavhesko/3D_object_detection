import open3d

from utils.data_utils import load_model


class Annotator:

    def __init__(self, input_folder, model_path):
        self._input_folder = input_folder
        self._model_path = model_path
        self._model = load_model(model_path)

    def execute(self):
        pass

    def _preprocess(self):
        pass

    def _postprocess(self):
        pass


if __name__ == "__main__":
    pass
