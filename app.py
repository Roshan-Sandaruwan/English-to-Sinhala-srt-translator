
import os
import re
from io import BytesIO

from flask import Flask, render_template, request, jsonify, send_file

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


app = Flask(__name__)

# Maximum upload size: 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


MODEL_NAME = os.getenv(
    "TRANSLATION_MODEL",
    "kawinduwijewardhane/nllb-sinhala-english"
)

TIME_RE = re.compile(
    r"^\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+"
    r"\d{2}:\d{2}:\d{2},\d{3}(?:\s+.*)?$"
)


_tokenizer = None
_model = None
_device = None


# ---------------------------------------------------------
# Load translation model
# ---------------------------------------------------------

def get_model():

    global _tokenizer, _model, _device

    if _model is None:

        print("Loading translation model...")

        _device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Using device: {_device}")

        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            src_lang="eng_Latn",
            # Older model metadata stores this as a list, while recent
            # Transformers expects a mapping when registering special tokens.
            extra_special_tokens={}
        )

        _model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME
        )

        _model.to(_device)
        _model.eval()

        print("Translation model loaded successfully.")

    return _tokenizer, _model, _device


# ---------------------------------------------------------
# Parse SRT
# ---------------------------------------------------------

def parse_srt(text):

    # Normalize line endings
    text = (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .lstrip("\ufeff")
    )

    # Split subtitle blocks using empty lines
    blocks = re.split(r"\n\s*\n", text.strip())

    items = []

    for block in blocks:

        lines = block.split("\n")

        if len(lines) < 2:
            continue

        time_i = None

        for i, line in enumerate(lines):
            if TIME_RE.match(line.strip()):
                time_i = i
                break

        if time_i is None:
            continue

        # Usually the subtitle number is before the timestamp
        if time_i > 0:
            index = lines[time_i - 1].strip()
        else:
            index = str(len(items) + 1)

        subtitle_text = "\n".join(
            lines[time_i + 1:]
        ).strip()

        if not subtitle_text:
            continue

        items.append({
            "index": index,
            "time": lines[time_i].strip(),
            "text": subtitle_text
        })

    return items


# ---------------------------------------------------------
# Build new SRT
# ---------------------------------------------------------

def build_srt(items):

    blocks = []

    for item in items:

        block = (
            f"{item['index']}\n"
            f"{item['time']}\n"
            f"{item['text']}"
        )

        blocks.append(block)

    return "\n\n".join(blocks) + "\n"


# ---------------------------------------------------------
# Translate English -> Sinhala
# ---------------------------------------------------------

def translate_texts(texts, batch_size=4):

    tokenizer, model, device = get_model()

    results = []

    # Sinhala language token
    target_id = tokenizer.convert_tokens_to_ids("sin_Sinh")

    if target_id is None:
        raise RuntimeError(
            "Sinhala language token 'sin_Sinh' was not found "
            "in the tokenizer."
        )

    for start in range(0, len(texts), batch_size):

        current_texts = texts[start:start + batch_size]

        cleaned_texts = []

        for text in current_texts:

            text = text.replace("\n", " ")
            text = re.sub(r"\s+", " ", text)
            text = text.strip()

            cleaned_texts.append(text)

        print(
            f"Translating subtitles "
            f"{start + 1} - "
            f"{min(start + batch_size, len(texts))} "
            f"of {len(texts)}"
        )

        encoded = tokenizer(
            cleaned_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )
        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        with torch.inference_mode():

            generated = model.generate(
                **encoded,
                forced_bos_token_id=target_id,
                max_new_tokens=256,
                num_beams=3
            )

        translated_texts = tokenizer.batch_decode(
            generated,
            skip_special_tokens=True
        )

        results.extend(
            translated_text.strip()
            for translated_text in translated_texts
        )

    return results


# ---------------------------------------------------------
# Home page
# ---------------------------------------------------------

@app.get("/")
def home():

    return render_template(
        "index.html",
        model_name=MODEL_NAME
    )


# ---------------------------------------------------------
# Translation endpoint
# ---------------------------------------------------------

@app.post("/translate")
def translate():

    uploaded_file = request.files.get("file")

    if not uploaded_file:

        return jsonify(
            error="Please select an SRT file."
        ), 400

    if not uploaded_file.filename.lower().endswith(".srt"):

        return jsonify(
            error="Please upload a valid .srt file."
        ), 400

    raw = uploaded_file.read()

    # Decode subtitle file
    try:

        text = raw.decode("utf-8-sig")

    except UnicodeDecodeError:

        text = raw.decode(
            "cp1252",
            errors="replace"
        )

    # Parse subtitles
    items = parse_srt(text)

    if not items:

        return jsonify(
            error="No valid subtitle entries were found."
        ), 400

    print(f"Found {len(items)} subtitle entries.")

    try:

        original_texts = [
            item["text"]
            for item in items
        ]

        translated_texts = translate_texts(
            original_texts
        )

        if len(translated_texts) != len(items):

            raise RuntimeError(
                "The number of translated subtitles "
                "does not match the original subtitles."
            )

        for item, translated_text in zip(
            items,
            translated_texts
        ):

            item["text"] = translated_text

    except Exception as e:

        print("Translation error:", repr(e))

        return jsonify(
            error=f"Local translation failed: {str(e)}"
        ), 500

    # Generate Sinhala SRT
    result = build_srt(items)

    result_bytes = result.encode("utf-8-sig")

    original_name = os.path.basename(
        uploaded_file.filename
    )

    base_name = os.path.splitext(
        original_name
    )[0]

    output_name = f"{base_name}_si.srt"

    print("Translation completed successfully.")

    return send_file(
        BytesIO(result_bytes),
        mimetype="application/x-subrip",
        as_attachment=True,
        download_name=output_name
    )


# ---------------------------------------------------------
# Start Flask
# ---------------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )
