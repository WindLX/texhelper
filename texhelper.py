import re
import io
import os
import csv

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

    def load_model(self, path):
        try:
            tokenizer = AutoTokenizer.from_pretrained(path)
            feature_extractor = AutoImageProcessor.from_pretrained(path)
            encoder_session = ort.InferenceSession(f"{path}/encoder_model.onnx")
            decoder_session = ort.InferenceSession(f"{path}/decoder_model_merged.onnx")
            print("\n===成功加载模型===\n")
        except Exception as e:
            raise IOError(f"Error loading models or tokenizer: {e}")
        return (tokenizer, feature_extractor, encoder_session, decoder_session)

    def mixtex_inference(
        self,
        image,
        max_length=512,
        num_layers=3,
        hidden_size=768,
        num_attention_heads=12,
        batch_size=1,
    ):
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

    def convert_align_to_equations(self, text):
        text = re.sub(r"\\begin\{align\*\}|\\end\{align\*\}", "", text).replace("&", "")
        equations = text.strip().split("\\\\")
        converted = []
        for eq in equations:
            eq = eq.strip().replace("\\[", "").replace("\\]", "").replace("\n", "")
            if eq:
                converted.append(f"$$ {eq} $$")
        return "\n".join(converted)

    def pad_image(self, img, out_size):
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


mixtex_app = MixTeXApp()


@app.post("/ocr/")
async def ocr(image: UploadFile = File(...)):
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
    run("texhelper:app", host="0.0.0.0", port=8000, reload=True)
