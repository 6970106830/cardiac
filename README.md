# Cardiac Triage Copilot

Capstone project T2, 3099704 AI for Digital Health, Faculty of Medicine, Chulalongkorn University (18 October 2026).

A resource-aware triage prototype for patients with suspected cardiac symptoms at triage or ER. The nurse sets up the hospital's available tests, enters the patient's history and vital signs, and gets a red / yellow / green triage level. The tool then lists the tests this hospital can do now and the cases that need a consult or a referral, and produces an A4 summary for the doctor on duty to sign.

> **Research prototype for teaching. Not for clinical use. Never enter real patient data.**

## Run it

- **Online:** enable GitHub Pages (Settings → Pages → Deploy from branch → `main` / root) and open the published URL.
- **Offline:** download `index.html` and open it in any browser. Everything runs in the browser; no server is needed.
- **Print the A4 sheet:** press "Evaluate", then "พิมพ์ใบสรุป (A4)" or Ctrl+P / ⌘P.
- **Notebook:** `notebook/T2_chest_pain_ruleout.ipynb` downloads the UCI data automatically if the `heartdisease/` folder is missing.

The "extract with Claude" button (free text or dictation → form fields) works only when the page is opened as a claude.ai artifact. On GitHub Pages it is hidden, and the form works without it.

## How the triage works

| Level | Rule |
|---|---|
| **Red** | Any red flag (syncope, SBP < 90 or ≥ 200, HR < 50 or > 120, chest pain with ST-T change, positive troponin) **or** model risk ≥ 70 % |
| **Green** | Model risk below the validated cut-off (13–16 %, sensitivity ≥ 95 % in development) **and** the clinical rule agrees (atypical or non-anginal pain, age < 55) **and** chest pain is the only complaint, with no ECG abnormality |
| **Yellow** | Everything else |

Model: three logistic regressions, picked by the data available. Tier A uses age, sex, chest-pain type and SBP; tier B adds the resting ECG; tier C adds cholesterol and FBS. All three were trained on 227 Cleveland patients and give AUROC ≈ 0.84 in CV, 0.83–0.84 on the sealed test set and 0.83–0.84 in Hungary (external). Risk drivers are linear SHAP values, which are exact for logistic regression on the log-odds scale.

## Known limitations

- The label is ≥ 50 % stenosis on angiography in patients **already referred** for it (UCI, 1980s). It is not an ACS label, so red is decided by clinical red-flag rules first.
- Referral bias: "no chest pain" (cp = 4) carries a high CAD rate in the training data, and typical angina a lower one. The tool warns about this, and red-flag rules catch typical angina with ECG changes.
- A 35 % green cut-off would drop sensitivity to 0.73–0.80 in Hungary, so the tool uses the validated cut-off plus the clinical rule (sensitivity 0.94 in Hungary).
- ST-T abnormality appears in only 1 training patient, so it is handled as a rule, not by the model.
- The model fails to transfer to sicker populations (VA 74 %, Switzerland 93 % CAD). It would need local validation before any use.
- Differential diagnoses are a rule-based list by presenting complaint and do not come from the model.

## Repository

```
index.html                         the tool (single file)
notebook/T2_chest_pain_ruleout.ipynb   audit, baselines, sealed test, subgroups, external validation, TwoKey
model/train_tiered_models.py        trains the tier A/B/C models used in index.html
model/models.json                   exported coefficients
```

Data: [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease), CC BY 4.0 (Janosi, Steinbrunn, Pfisterer, Detrano, 1988).

AI-assistant use: Claude (Anthropic) helped write code and text. The group reviewed, ran and is responsible for every result.
