import os
import json
import numpy as np
import triton_python_backend_utils as pb_utils
import shutil


class TritonPythonModel:

    def execute(self, requests):
        responses = []
        for request in requests:
            all_logits = pb_utils.get_input_tensor_by_name(
                request, "PREDICTION_FP16"
            ).as_numpy()
            threshold = pb_utils.get_input_tensor_by_name(
                request, "THRESHOLD"
            ).as_numpy()[0]
            score = np.exp(all_logits[:, 1]) / np.sum(np.exp(all_logits), axis=1)

            score_tensor = pb_utils.Tensor(
                "score_fp16", np.array(score, dtype=np.float32)
            )
            verdict_tensor = pb_utils.Tensor(
                "verdict_fp16", np.array(score > threshold, dtype=np.bool_)
            )
            response = pb_utils.InferenceResponse(
                output_tensors=(score_tensor, verdict_tensor)
            )
            responses.append(response)
        return responses
