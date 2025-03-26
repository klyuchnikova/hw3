import os
import json
import numpy as np
import triton_python_backend_utils as pb_utils
import hashlib
from utils import preproc_images
import shutil


class TritonPythonModel:

    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        self.fp16_output_name = "IMAGE_FP16"

    def execute(self, requests):
        responses = []
        for request in requests:
            image_filenames = [
                text[0].decode("utf-8")
                for text in pb_utils.get_input_tensor_by_name(
                    request, "IMAGE_PATH"
                ).as_numpy()
            ]

            preprocessed_images = preproc_images(image_filenames, target_size=224)

            preprocessed_images_np = np.array(preprocessed_images, dtype=np.float32)
            images_tensor_fp16 = pb_utils.Tensor(
                self.fp16_output_name, preprocessed_images_np.astype(np.float16)
            )
            response = pb_utils.InferenceResponse(output_tensors=[images_tensor_fp16])
            responses.append(response)
        return responses
