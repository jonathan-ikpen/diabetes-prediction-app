# Diabetes Risk Prediction Web App

A Flask web app that estimates a person's risk of diabetes from 8 clinical
measurements, then explains the result in plain English.

It uses two kinds of AI together:

- **Machine Learning** (a Random Forest model) works out the **number** — the risk percentage.
- **An LLM** (Groq, running `openai/gpt-oss-120b`) turns that number into a **readable explanation**.

The LLM is never allowed to invent medical facts. Every WHO/IDF threshold it
mentions is handed to it from a file in this project. If the internet or the API
key is missing, the app writes the explanation itself from that same file — so
it always works.

---

## How to Run

You need **Python 3.9 or newer**.

### 1. Open a terminal in this folder

### 2. Create a virtual environment and activate it

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
```

### 3. Install the packages

```bash
pip install -r requirements.txt
```

### 4. Start the app

```bash
python run.py
```

### 5. Open it in your browser

**<http://127.0.0.1:8137>**

That's it. The trained model is already included, so you do **not** need to
train anything, and you do **not** need an API key.

### Optional: turn on AI explanations

Without a key the app still explains every result — it just writes the
explanation itself instead of asking an LLM. To use Groq:

1. Get a free API key at **<https://console.groq.com/keys>**
2. Copy `.env.example` to `.env`:
   ```bash
   copy .env.example .env       # Windows
   cp .env.example .env         # macOS / Linux
   ```
3. Open `.env` and paste your key:
   ```
   LLM_API_KEY=gsk_your_key_here
   ```
4. Restart the app.

### Optional: retrain the model yourself

```bash
python -m training.fetch_dataset    # download and check the dataset
python -m training.preprocess       # clean it
python -m training.train            # train and save the model
python -m training.evaluate         # check how good the saved model is
```

### Run the tests

```bash
python -m unittest discover -s tests
```

> **If the app won't start** and you see *"An attempt was made to access a socket
> in a way forbidden by its access permissions"*, that port is already taken.
> Open `.env` and change `FLASK_PORT` to another number, like `8145`.

---

## Stack

- Python 3.9+
- Flask (web framework)
- Scikit-Learn (Random Forest model)
- Pandas & NumPy (data handling)
- Joblib (saving the trained model)
- Groq API — `openai/gpt-oss-120b` (AI explanations, free)
- HTML5
- Vanilla CSS3 (hand-written, no Bootstrap or Tailwind)
- Vanilla JavaScript (no React, no build step)
- Anton & Inter fonts (included in the project, so no internet needed)

---

## Main Folders

There are only **3 folders** at the top level. Each has one job.

| Folder | What it is for |
|---|---|
| `app` | **The website.** Everything that runs when someone visits a page |
| `training` | **How the model was made.** The scripts, the data, and the finished model |
| `tests` | **Proof the code works.** Automated checks that run in seconds |

Plus one file that starts everything: **`run.py`**

> **What `tests` is — and is not.** It is *not* the model's accuracy results.
> It is a set of automated checks that run the code and confirm it still
> behaves: pages load, bad input is rejected, security headers are present,
> the model gives the same answer twice, and the dataset has no leakage. Think
> of it as a checklist a robot runs for you in 3 seconds.
>
> **The model's accuracy results live in `training/model/evaluation_report.json`**,
> and are displayed on the **Model Card** page at `/model-card`.

### Inside `app` — the website

The website is split into 4 layers. Each layer only talks to the one below it.

| Folder | What it is for |
|---|---|
| `app/interface` | What the user sees — pages, styling, buttons, forms |
| `app/engine` | The thinking — checks the input, runs the model, writes the explanation |
| `app/knowledgebase` | The medical facts — every WHO/IDF number lives here |
| `app/config` | Settings — reads the `.env` file, sets security headers |

### Inside `training` — how the model was made

| Folder / File | What it is for |
|---|---|
| `training/fetch_dataset.py` | Downloads the dataset and checks it isn't corrupted |
| `training/preprocess.py` | Cleans the dataset |
| `training/train.py` | Trains the model and saves it |
| `training/evaluate.py` | Reports how accurate the saved model is |
| `training/data` | The dataset, before and after cleaning |
| `training/model` | The finished model (`model.pkl`) and its score reports |

### Full layout

```
diabetes-app/
├── run.py                    START HERE - runs the website
├── requirements.txt          list of packages to install
├── .env.example              settings template (copy this to .env)
├── README.md                 this file
│
├── app/                      THE WEBSITE
│   ├── interface/            what the user sees
│   │   ├── routes.py         which page shows for which web address
│   │   ├── api.py            JSON endpoints used by the JavaScript
│   │   ├── templates/        the HTML pages
│   │   └── static/           the CSS, JavaScript and fonts
│   │
│   ├── engine/               the thinking
│   │   ├── validation.py     checks the 8 numbers are sensible
│   │   ├── predictor.py      runs the model, gets the risk %
│   │   ├── interview.py      turns plain answers into the 8 numbers
│   │   ├── narrative.py      asks Groq to explain it
│   │   └── chat.py           the floating assistant
│   │
│   ├── knowledgebase/        the medical facts
│   │   ├── who_idf_thresholds.json   every WHO/IDF number
│   │   ├── clinical_reference.py     reads that file
│   │   └── prompt_builder.py         builds the LLM's instructions
│   │
│   └── config/settings.py    app settings and security
│
├── training/                 HOW THE MODEL WAS MADE
│   ├── fetch_dataset.py      downloads + checks the dataset
│   ├── preprocess.py         cleans the dataset
│   ├── train.py              trains the model, saves model.pkl
│   ├── evaluate.py           tests how good the saved model is
│   ├── data/
│   │   ├── raw/              the original dataset
│   │   └── processed/        the cleaned dataset
│   └── model/
│       ├── model.pkl         the trained model
│       ├── model_metadata.json     what settings it was trained with
│       └── evaluation_report.json  how accurate it is
│
└── tests/test_app.py         76 automated checks (not model results)
```

---

## The Pages

| Address | Page |
|---|---|
| `/` | Home — what the tool is and how it works |
| `/assess` | The form where you enter the 8 measurements |
| `/result` | Your risk score and the explanation |
| `/clinical-guidelines` | All the WHO/IDF numbers the app uses |
| `/model-card` | How accurate the model is, and where it gets things wrong |
| `/about` | Project background and design decisions |
| `/health` | Live AI status check (JSON) |
| `/api/health` | Full system status: model + AI + knowledge base (JSON) |
| `/api/chat` | The floating assistant (JSON, POST) |

There is also a **floating assistant** in the bottom-right corner of every page.
You can describe yourself in your own words &mdash; *"I'm 47, 102kg, 1.7m tall, my
mum and brother have diabetes"* &mdash; and it works out the rest.

> **The assistant never invents the risk number.** It reads your message, pulls
> out the values, and then calls the same Random Forest the other two routes
> use. Only then does it explain what came back. A chatbot that estimated risk
> by itself would sound completely convincing and be based on nothing.
>
> It also refuses to name medicines, refuses to diagnose, and tells you to seek
> urgent care if you describe an emergency. The launcher does not appear at all
> unless an API key is set, because an assistant that cannot answer is worse
> than no assistant.

---

## Checking the AI Is Working

Visit **<http://127.0.0.1:8137/health>** at any time:

```json
{"status":"operational","llm":"connected","model":"openai/gpt-oss-120b","provider":"groq"}
```

This is a live ping to Groq, not just a settings read. It catches an expired
key, an exhausted quota, or a model Groq has retired — problems that would
otherwise silently downgrade every explanation to the offline writer without
telling you.

`llm` will be one of:

| Value | Meaning | What to do |
|---|---|---|
| `connected` | Working normally | Nothing |
| `not_configured` | No API key set | Add `LLM_API_KEY` to `.env` |
| `unauthorized` | Key rejected, wrong or revoked | Get a new key from the Groq console |
| `rate_limited` | Free-tier quota used up | Wait a minute, or upgrade |
| `model_not_found` | Groq retired the model | Set `LLM_MODEL` to one it still offers |
| `unreachable` | No internet / DNS failure | Check the connection |
| `disabled` | Turned off on purpose | Set `LLM_ENABLED=true` |
| `provider_error` | Groq is having an outage | Wait |

It returns HTTP **200** when connected and **503** otherwise, so an uptime
monitor can watch it directly. Add `?force=1` to skip the 30-second cache.

For a fuller report (the ML model *and* the AI *and* the knowledge base), use
**`/api/health`**.

> **A note on the free tier.** Groq's free plan allows about **8,000 tokens per
> minute**, and each explanation uses roughly 1,200. So around 6 screenings a
> minute get an AI-written explanation; beyond that the app quietly falls back
> to its own writer and carries on. You will see `rate_limited` at `/health`
> when this happens. Nothing breaks.

---

## How This Was Built, Step by Step

The whole build, in plain English.

### Step 1 — Got the data, and found a problem

The project needed a dataset of real patients. The one supplied was the **Pima
Indians Diabetes Dataset** (768 patients, 8 measurements each, plus whether they
actually had diabetes).

But that copy was **broken in a way that is easy to miss**. Someone had already
"cleaned" it by filling in missing values — but they filled them in *differently
depending on the answer*:

- Every patient with `Insulin = 102.5` did **not** have diabetes. All 236 of them.
- Every patient with `Insulin = 169.5` **did** have diabetes. All 138 of them.

That is 374 patients — **half the dataset** — where the Insulin column secretly
spelled out the answer. This is called **data leakage**.

A model trained on it scored **95% accuracy**, which looks fantastic but is fake.
It wasn't learning medicine; it was reading the answer key.

**Fix:** downloaded the original untouched dataset, and wrote
`training/fetch_dataset.py`, which refuses to accept any dataset with this
problem. Your original file is still kept at
`training/data/raw/pima_preimputed_LEAKY_do_not_train.csv` so you can see it.

### Step 2 — Cleaned the data properly

In the real dataset, missing measurements are written as `0`. But a living
person cannot have a blood glucose of 0. So `0` really means "not measured".

`training/preprocess.py` replaces those zeros with the middle value (the median)
of that column. Importantly, this happens **inside** the model, using only the
training patients — so the model can never peek at the test patients.

### Step 3 — Trained the model

`training/train.py` does this:

1. Splits the 768 patients: 614 to learn from, 154 hidden away for testing.
2. Tries **72 different model settings** and keeps the best one.
3. Trains a **Random Forest** — imagine 200 decision trees each voting on
   whether this person is at risk, then averaging the votes.
4. Tests it on the 154 hidden patients it has never seen.
5. Saves the finished model to `training/model/model.pkl`.

### Step 4 — Made it safer for screening

By default a model says "at risk" when it is more than **50%** sure. At that
setting, this model **missed 25 of the 54 patients who actually had diabetes**.

For a screening tool that is the wrong trade-off:

- Wrongly telling a healthy person to get a blood test → mild inconvenience.
- Wrongly telling a sick person they're fine → they go untreated.

So the cut-off was lowered to **34.3%**. Missed patients dropped from **25 to 12**.
The app pays for this with more false alarms, on purpose.

### Step 5 — Built the medical knowledge base

All the WHO and IDF medical numbers went into **one file**:
`app/knowledgebase/who_idf_thresholds.json`.

For example, glucose:
- Below 140 = normal
- 140–199 = prediabetes
- 200 or above = diabetes range

Everything reads from this one file — the form's allowed ranges, the results
page, and the AI's instructions. Change a number there and it changes
everywhere. Nothing is hardcoded anywhere else.

### Step 6 — Built the inference engine

Three small files that do the actual work:

- `validation.py` — rejects anything that isn't a sensible number (letters,
  negative ages, a glucose of 5000, and so on)
- `predictor.py` — loads the trained model and produces the risk percentage
- `narrative.py` — asks the LLM for the explanation

### Step 7 — Connected the LLM (Groq)

The app sends Groq the model's output **plus the exact WHO facts it is allowed
to use**, and tells it: *only use these facts, never diagnose, write three short
paragraphs.*

The LLM is a **writer, not a doctor**. It rephrases facts it was handed. This
avoids the biggest danger with medical AI: a model confidently inventing a wrong
medical threshold from memory.

**Backup plan:** if there's no API key, no internet, or the API fails, the app
writes the same three paragraphs itself from the knowledge base. The user always
gets an explanation.

A `/health` endpoint was added so you can tell at a glance whether the AI is
actually working, instead of guessing why an explanation looks plainer than
usual.

### Step 8 — Built the interface

6 pages, hand-written HTML and CSS, no frameworks. The design rules:

- Flat surfaces — no shadows, no gradients
- Everything separated by thin 1px lines
- `Anton` font for headings, `Inter` for text
- Pill-shaped buttons and inputs
- Light and dark mode
- Fonts bundled in the project, so it works offline

### Step 9 — Tested it properly

Every page was loaded in a real browser at phone, tablet and desktop sizes. That
caught 4 bugs that reading the code would never reveal — including that **every
progress bar was invisible**, because of a small CSS mistake.

Then 76 automated tests were written, covering input checking, the medical
thresholds, the model, and the data-leakage guard.

---

## How Accurate Is It?

Tested on 154 patients the model had never seen:

| Measure | Score | What it means |
|---|---|---|
| ROC-AUC | **0.809** | Overall skill. 0.5 = guessing, 1.0 = perfect |
| Sensitivity | **78%** | Catches 78 out of 100 people who really are at risk |
| Specificity | **70%** | Correctly clears 70 out of 100 healthy people |
| Accuracy | **73%** | Right about 73 times out of 100 |

Which measurements matter most to the model:

1. **Glucose — 37%** (correct: this is what doctors use to define diabetes)
2. BMI — 16%
3. Age — 13%

---

## Security

- Every input is checked before it reaches the model
- API keys live in `.env`, which is never committed to Git
- Security headers on every page (CSP, no-sniff, clickjacking protection)
- **Nothing is stored.** No database, no cookies holding health data. Your
  measurements are used, then thrown away

---

## Important Limitations

**This is not a medical diagnosis.** Please read these:

1. **It cannot diagnose diabetes.** Only a lab blood test can. This tool tells
   you whether it's worth getting that test.
2. **The training patients are not Nigerian.** The dataset is adult women of
   Pima Indian heritage in the USA. Risk patterns differ. Treat the percentage
   as a nudge to get tested, not a personal prediction.
3. **It's a small dataset.** 768 patients is not many, so the accuracy figures
   carry real uncertainty.
4. **It was trained only on women.** The "number of pregnancies" input means
   nothing for men.

---

*Built for educational and early-detection purposes. Not a medical device.*
