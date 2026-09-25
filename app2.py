import os, re
from io import BytesIO
from flask import Flask, render_template, request, jsonify, send_file
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

MODEL_NAME = os.getenv('TRANSLATION_MODEL', 'kawinduwijewardhane/nllb-sinhala-english')
TIME_RE = re.compile(r'^\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}(?:\s+.*)?$')
_tokenizer = None
_model = None
_device = None


def get_model():
    global _tokenizer, _model, _device
    if _model is None:
        _device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang='eng_Latn')
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        _model.to(_device)
        _model.eval()
    return _tokenizer, _model, _device


def parse_srt(text):
    text = text.replace('\r\n', '\n').replace('\r', '\n').lstrip('\ufeff')
    blocks = re.split(r'\n\s*\n', text.strip())
    items = []
    for block in blocks:
        lines = block.split('\n')
        if len(lines) < 2:
            continue
        idx = lines[0].strip()
        time_i = 1 if len(lines) > 1 and TIME_RE.match(lines[1].strip()) else next(
            (i for i, line in enumerate(lines) if TIME_RE.match(line.strip())), None
        )
        if time_i is None:
            continue
        items.append({
            'index': idx,
            'time': lines[time_i].strip(),
            'text': '\n'.join(lines[time_i + 1:]).strip()
        })
    return items


def build_srt(items):
    return '\n\n'.join(f"{x['index']}\n{x['time']}\n{x['text']}" for x in items) + '\n'


def translate_texts(texts, batch_size=8):
    tokenizer, model, device = get_model()
    results = []
    target_id = tokenizer.convert_tokens_to_ids('sin_Sinh')

    for start in range(0, len(texts), batch_size):
        batch = [t.replace('\n', ' ').strip() for t in texts[start:start + batch_size]]
        encoded = tokenizer(
            batch,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=256
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                forced_bos_token_id=target_id,
                max_new_tokens=256,
                num_beams=3
            )
        results.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return results


@app.get('/')
def home():
    return render_template('index.html', model_name=MODEL_NAME)


@app.post('/translate')
def translate():
    f = request.files.get('file')
    if not f or not f.filename.lower().endswith('.srt'):
        return jsonify(error='Please upload a valid .srt file.'), 400

    raw = f.read()
    try:
        text = raw.decode('utf-8-sig')
    except UnicodeDecodeError:
        text = raw.decode('cp1252', errors='replace')

    items = parse_srt(text)
    if not items:
        return jsonify(error='No valid subtitle entries were found.'), 400

    try:
        translated = translate_texts([x['text'] for x in items])
        for item, new_text in zip(items, translated):
            item['text'] = new_text
    except Exception as e:
        return jsonify(error=f'Local translation failed: {e}'), 500

    result = build_srt(items).encode('utf-8-sig')
    base = os.path.splitext(os.path.basename(f.filename))[0]
    return send_file(
        BytesIO(result),
        mimetype='application/x-subrip; charset=utf-8',
        as_attachment=True,
        download_name=f'{base}_si.srt'
    )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
