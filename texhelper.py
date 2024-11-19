import re
import io
import os
import logging

import numpy as np
import onnxruntime as ort
from transformers import AutoImageProcessor, AutoTokenizer
from PIL import Image

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from uvicorn import run
from fastapi import Depends

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


class MixTeXApp:
    def __init__(self):
        self.data_folder = "data"
        self.use_dollars_for_inline_math = False
        self.convert_align_to_equations_enabled = False
        self.ocr_paused = False
        self.current_image = None
        self.output = None
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
        self.model = self.load_model("onnx")

    def load_model(self, path: str):
        """
        Load the tokenizer, feature extractor, and ONNX models from the specified path.

        Args:
            path (str): The file path to the directory containing the model files.

        Returns:
            tuple: A tuple containing the following elements:
            - tokenizer (AutoTokenizer): The tokenizer loaded from the specified path.
            - feature_extractor (AutoImageProcessor): The feature extractor loaded from the specified path.
            - encoder_session (ort.InferenceSession): The ONNX inference session for the encoder model.
            - decoder_session (ort.InferenceSession): The ONNX inference session for the decoder model.

        Raises:
            IOError: If there is an error loading the models or tokenizer.
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(path)
            feature_extractor = AutoImageProcessor.from_pretrained(path)
            encoder_session = ort.InferenceSession(f"{path}/encoder_model.onnx")
            decoder_session = ort.InferenceSession(f"{path}/decoder_model_merged.onnx")
            logging.info("Successfully loaded models and tokenizer")
        except Exception as e:
            raise IOError(f"Error loading models or tokenizer: {e}")
        return (tokenizer, feature_extractor, encoder_session, decoder_session)

    def mixtex_inference(
        self,
        image,
        max_length: int = 512,
        num_layers: int = 3,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        batch_size: int = 1,
    ) -> str:
        tokenizer, feature_extractor, encoder_session, decoder_session = self.model
        try:
            generated_text = ""
            head_size = hidden_size // num_attention_heads
            inputs = feature_extractor(image, return_tensors="np").pixel_values
            encoder_outputs = encoder_session.run(None, {"pixel_values": inputs})[0]
            decoder_inputs = {
                "input_ids": tokenizer("<s>", return_tensors="np").input_ids.astype(
                    np.int64
                ),
                "encoder_hidden_states": encoder_outputs,
                "use_cache_branch": np.array([True], dtype=bool),
                **{
                    f"past_key_values.{i}.{t}": np.zeros(
                        (batch_size, num_attention_heads, 0, head_size),
                        dtype=np.float32,
                    )
                    for i in range(num_layers)
                    for t in ["key", "value"]
                },
            }
            for _ in range(max_length):
                decoder_outputs = decoder_session.run(None, decoder_inputs)
                next_token_id = np.argmax(decoder_outputs[0][:, -1, :], axis=-1)
                generated_text += tokenizer.decode(
                    next_token_id, skip_special_tokens=True
                )
                if next_token_id == tokenizer.eos_token_id:
                    break
                decoder_inputs.update(
                    {
                        "input_ids": next_token_id[:, None],
                        **{
                            f"past_key_values.{i}.{t}": decoder_outputs[i * 2 + 1 + j]
                            for i in range(num_layers)
                            for j, t in enumerate(["key", "value"])
                        },
                    }
                )
            if self.convert_align_to_equations_enabled:
                generated_text = self.convert_align_to_equations(generated_text)
            return generated_text
        except Exception as e:
            raise RuntimeError(f"Error during OCR: {e}")

    @staticmethod
    def convert_align_to_equations(text: str) -> str:
        text = re.sub(r"\\begin\{align\*\}|\\end\{align\*\}", "", text).replace("&", "")
        equations = text.strip().split("\\\\")
        converted = []
        for eq in equations:
            eq = eq.strip().replace("\\[", "").replace("\\]", "").replace("\n", "")
            if eq:
                converted.append(f"$$ {eq} $$")
        return "\n".join(converted)

    @staticmethod
    def pad_image(img, out_size: tuple[int, int]):
        x_img, y_img = out_size
        background = Image.new("RGB", (x_img, y_img), (255, 255, 255))
        width, height = img.size
        if width < x_img and height < y_img:
            x = (x_img - width) // 2
            y = (y_img - height) // 2
            background.paste(img, (x, y))
        else:
            scale = min(x_img / width, y_img / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img_resized = img.resize((new_width, new_height), Image.LANCZOS)
            x = (x_img - new_width) // 2
            y = (y_img - new_height) // 2
            background.paste(img_resized, (x, y))
        return background


async def get_mixtex():
    mixtex_app = MixTeXApp()
    try:
        yield mixtex_app
    finally:
        pass


@app.post("/ocr/")
async def ocr(
    image: UploadFile = File(...), mixtex_app: MixTeXApp = Depends(get_mixtex)
):
    try:
        image = Image.open(io.BytesIO(await image.read())).convert("RGB")
        image = mixtex_app.pad_image(image, (448, 448))
        result = mixtex_app.mixtex_inference(image)
        result = (
            result.replace("\\[", "\\begin{align*}")
            .replace("\\]", "\\end{align*}")
            .replace("%", "\\%")
        )
        if mixtex_app.use_dollars_for_inline_math:
            result = result.replace("\\(", "$").replace("\\)", "$")
        return JSONResponse(content={"text": result})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    run("texhelper:app", host="0.0.0.0", port=8000, reload=True)
