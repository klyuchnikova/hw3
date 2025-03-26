import os
import json
import numpy as np
import triton_python_backend_utils as pb_utils
import hashlib
from utils import preprocess_image
import shutil


class TritonPythonModel:

    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        self.fp16_output_name = "IMAGE_FP16"

    def execute(self, requests):
        responses = []
        for request in requests:
            image_data = pb_utils.get_input_tensor_by_name(
                request, "IMAGE_DATA"
            ).as_numpy()

            preprocessed_images = []
            for img_array in image_data:
                processed = preprocess_image(img_array)
                preprocessed_images.append(processed)

            preprocessed_images_np = np.stack(preprocessed_images)
            images_tensor_fp16 = pb_utils.Tensor(
                self.fp16_output_name, preprocessed_images_np.astype(np.float16)
            )
            responses.append(
                pb_utils.InferenceResponse(output_tensors=[images_tensor_fp16])
            )
        return responses
