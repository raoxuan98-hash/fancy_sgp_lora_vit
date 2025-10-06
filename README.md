# sgp_lora_vit

SGP + LoRA + ViT project.

## Quick Start

1. Create and activate virtual env
   - python -m venv .venv
   - .venv\\Scripts\\activate
2. Install deps
   - pip install -r requirements.txt
3. Run training or scripts
   - python train.py

## Notes

- Large artifacts and logs are ignored by .gitignore (datasets/, sldc_logs_authors/, model checkpoints, etc.).
- Add minimal sample data under data/sample if needed and whitelist with a negation rule.