# Free Local English → Sinhala SRT Translator

This version does **not** use the OpenAI API and needs no API key or paid credits.
It runs a Sinhala/English NLLB translation model locally on your computer.

## Requirements
- Python 3.10 or 3.11 recommended
- Internet connection for the FIRST run only, so the model can download
- Several GB of free disk space
- 8 GB RAM minimum recommended; 16 GB is better
- NVIDIA GPU is optional. CPU works but is slower.

## Windows setup
Open Command Prompt in this folder:

    python -m venv venv
    venv\Scripts\activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    python app.py

Then open:

    http://127.0.0.1:5000

## First run
The first translation downloads the model from Hugging Face. This is a one-time download.
After it is cached, translation can run locally without API credits.

## Notes
- Upload an English .srt file.
- Subtitle timestamps and numbering are preserved.
- Output is UTF-8 Sinhala .srt.
- Local machine translation can be less natural than a large language model.
- The included model is CC BY-NC 4.0 / non-commercial. Review its license before commercial deployment.
