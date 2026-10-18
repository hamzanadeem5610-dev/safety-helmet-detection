# Setup & Run Guide — Safety Helmet Detection

Everything runs in Google Colab; you only need a Google account, a free Roboflow account and (only if asked) a free Kaggle account.

## 0. Put the project on Google Drive
1. Unzip `cv_project.zip` on your computer.
2. Open [drive.google.com](https://drive.google.com) → **My Drive** → *New → Folder upload* → choose the `cv_project` folder.
   The result must be exactly `My Drive/cv_project/` (the notebook expects this path).
3. In Drive, open `cv_project/notebooks/training_notebook.ipynb` → *Open with → Google Colaboratory*.
   Because the notebook lives on Drive, Colab auto-saves it **with all outputs** — that is your "notebook with visible outputs" deliverable.

## 1. First run (≈ 15 min, Day 1–2)
1. **Runtime → Change runtime type → T4 GPU → Save.**
2. Run cells from the top down to the end of **Phase 3 – "Prepare the manual-annotation batch"**.
   - Approve the Google Drive permission pop-up.
   - The dataset downloads with `kagglehub`. If it asks for credentials: kaggle.com → *Settings → API → Create New Token*, then upload the downloaded `kaggle.json` when the cell prompts.
3. Check the printed statistics and plots (duplicates removed, class balance, diversity) — they go in your report.

## 2. Manual annotation in Roboflow (Day 3–4, ≈ 2–3 hours)
1. Download `My Drive/cv_project/manual_annotations/manual_batch.zip` and unzip it (120 images).
2. [app.roboflow.com](https://app.roboflow.com) → *Create New Project* → **Object Detection** → name: `safety-helmet-detection`.
3. Upload the 120 images → *Save and Continue* → assign to yourself → **Start Annotating**.
4. Create exactly three classes: **`helmet`**, **`no_helmet`**, **`person`**. Follow `docs/ANNOTATION_GUIDELINES.md`.
   Shortcuts: `B` = box tool, `←/→` = previous/next image.
5. *Add images to Dataset* (any split is fine — the notebook does its own 70/20/10 split).
6. *Generate* a version: **Preprocessing: Auto-Orient only (remove Resize). Augmentation: none.**
7. *Export Dataset → Format: YOLOv8 → Download zip to computer.*
8. Rename the zip to **`roboflow_export.zip`** and upload it to `My Drive/cv_project/manual_annotations/`.

Optional bonus: add some of your **own photos** (e.g. a construction site or workshop near you) to the same Roboflow project before exporting — the notebook imports them automatically as self-collected data.

> Tip: you can run the whole notebook once *before* annotating to test it (it uses the public labels and warns). Then re-run from the **"Import my Roboflow annotations"** cell after uploading your export.

## 3. Training (Day 5, ≈ 30–45 min on T4)
Run Phase 4 → Phase 7. The YOLOv8n baseline (≈ 10–15 min) and YOLOv8s experiment (≈ 15–25 min) train for 50 epochs each.
If Colab disconnects, all completed runs are already on Drive; just re-run the setup cells, then continue.

## 4. Evaluation, inference, error analysis (Day 6)
- Run Phase 8 → Phase 10.
- **Video (recommended):** download a free worksite clip (e.g. search "construction workers" on pexels.com → free license), upload it to `My Drive/cv_project/videos/` before running the video cell. Without one, a demo video is assembled from test images.
- **Your own images (optional):** put photos in `My Drive/cv_project/unseen_images/`.
- **Manual review:** look at the 12 error cases displayed, then type your own explanation for each in `MANUAL_REVIEW_NOTES` and re-run that cell. This is the brief's "manually review at least 10 incorrect predictions".

## 5. Report, README, LinkedIn, demo (Day 7)
- Run the Final Deliverables cells → `report/final_report.pdf`, `report/linkedin_portfolio_report.md`, README results table, and the ✅/❌ checklist.
- **Rewrite the overview and "What I learned" sections in your own words** before posting (the brief asks for this).
- Run the last cell (Gradio) → it prints a public `https://….gradio.live` link — take screenshots for your portfolio.
- *File → Save* the notebook. Download the Drive folder or push it to GitHub (below).

## 6. GitHub (optional but recommended)
`best.pt` (~6–22 MB) and the 500-image dataset (~25 MB as JPEG) fit in a normal GitHub repo.
In a new Colab cell:
```bash
!cd /content/drive/MyDrive/cv_project && git init && git add . && git commit -m "Safety helmet detection - YOLOv8"
```
then create an empty repo on github.com and follow its "push an existing repository" commands (use a personal access token as the password).

## 7. Hugging Face Spaces demo (optional)
Create a new **Gradio** Space, upload `app/gradio_app.py` renamed to `app.py`, `models/best.pt` (as `best.pt`), and `app/requirements_space.txt` renamed to `requirements.txt`. The Space builds and gives you a public demo URL for LinkedIn.

## Troubleshooting
| Problem | Fix |
|---|---|
| `GPU Available: False` | Runtime → Change runtime type → T4 GPU, then *Runtime → Restart session* |
| CUDA out of memory | Set `"batch": 8` in `EXPERIMENTS` (config cell) |
| Colab usage limit reached | Set `RUN_EXPERIMENT_2 = False` (baseline alone meets the brief) or try again later |
| Roboflow import mapped a class to IGNORED | Rename that class in Roboflow to `helmet` / `no_helmet` / `person`, re-export |
| Annotation audit agreement very low (<20%) | You exported with Resize "Fit (letterbox)"; re-export with Auto-Orient only |
