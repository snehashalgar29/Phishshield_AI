# PhishShieldAI - Full package (BERT-ready) with demo model

## Quick start (local)

1. Create virtual env and install requirements:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\\Scripts\\activate
   pip install -r requirements.txt
   ```

2. Train the lightweight demo model (fast):
   ```bash
   python training/train_demo.py
   ```
   This creates `models/demo_model.joblib` used by the API as a fallback.

3. (Optional) Fine-tune BERT (requires internet & GPU):
   ```bash
   python training/train_bert.py --dataset data/phish_dataset.csv --output_dir models/bert_phish --epochs 3
   ```

4. Run the API:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

5. Test endpoints at http://127.0.0.1:8000/docs

## Notes
- The repo is BERT-ready; the demo model allows you to run everything locally without large downloads.
- After you train BERT, the API will automatically use it.
