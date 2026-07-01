# Deploy & Submit — Step by Step

## 1. Push code to GitHub
1. Create a new repo on github.com (e.g. `shl-assessment-recommender`), public or private is fine.
2. In this project folder, run:
   ```
   git init
   git add .
   git commit -m "SHL assessment recommender"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```

## 2. Deploy on Render
1. Go to render.com → New → Web Service.
2. Connect your GitHub repo.
3. Settings:
   - Environment: Python 3
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Add environment variable: `GROQ_API_KEY` = (your key from console.groq.com)
5. Click Create Web Service. First deploy takes a few minutes (installs sentence-transformers/faiss).

## 3. Verify it's live
Once deployed, Render gives you a URL like `https://shl-assessment-recommender.onrender.com`.
- Visit `https://<your-url>/health` → should show `{"status":"ok"}`
- Visit `https://<your-url>/docs` → interactive API docs, try `/chat` manually with a sample message

## 4. Run the eval harness (recommended before submitting)
On your own machine (needs Python + `requests` installed):
```
pip install requests
python tests/run_eval.py --url https://<your-render-url>
```
This replays all 10 sample traces and prints Recall@10 per trace + the mean. If any trace errors out, fix that first — it likely means a bug the automated grader would also hit.

## 5. Submit
Via the form link from SHL's email:
- Public API endpoint URL: your Render URL (the base URL, they'll hit /health and /chat themselves)
- Approach document: `APPROACH.md` in this folder — copy its content into Google Docs / Word, export as PDF, upload.

## What I (Claude) could not do for you
- Could not create your GitHub/Render/Groq accounts or click deploy — needs your login
- Could not run the live eval harness myself since it needs your deployed URL + your API key
- Could not test the real embedding model locally (my sandbox has no internet access to huggingface.co) — it will download automatically on Render's first build, since Render has full internet access. If for any reason it fails there too, the service still works correctly via the automatic TF-IDF fallback — just re-run the eval harness to confirm the Recall@10 you get either way.
