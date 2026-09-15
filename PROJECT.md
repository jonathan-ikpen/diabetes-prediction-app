# Diabetes Risk Prediction Web App

## Project Objective

Build a complete, production-quality **Diabetes Risk Prediction Web App** using a hybrid AI architecture (Machine Learning + LLM) with a **Flask** backend. This system will serve as a lightweight expert system for early diabetes detection in Nigeria, processing 8 core clinical parameters to generate a mathematical risk probability and a natural-language clinical narrative. This is a real software project, not a demo or code snippets.

## Core Requirements

- One continuous codebase.
- Implement an end-to-end Machine Learning pipeline (training the model from a dataset and saving it).
- Never regenerate previous modules; extend the existing project.
- Final deliverable: complete ZIP with source, trained model file, assets, and documentation.

## Technology

- Python 3.9+
- Flask
- Scikit-Learn (Random Forest Classifier)
- Pandas & NumPy
- Joblib (Model Serialization)
- HTML5
- Vanilla CSS3 (Strictly custom styling, NO frameworks)
- Vanilla JavaScript
- LLM API (e.g., Gemini or OpenAI) for narrative generation

## UI / UX Design System Constraints

**Aesthetic Goal**: The UI must be a minimalist, premium, handcrafted interface heavily inspired by Notion, Linear, and Vercel.

**STRICT "DO NOT" RULES (Zero Exceptions)**:
- NO box-shadows (surfaces must remain entirely flat).
- NO gradients (use solid colors exclusively).
- NO glassmorphism, blurring, or translucent overlays.
- NO CSS frameworks (Do NOT use Bootstrap, Tailwind, or Material UI).
- NO heavy or dark borders for layout boundaries.
- NO "weird grey" backgrounds in light mode. The application canvas MUST be pure white (`#ffffff`).
- NO "kissing" components. Never let layout elements visually touch; ensure generous margins and grid gaps (e.g., `gap: 32px`).
- NO horizontal scrolling for list/board data. Use CSS Grid `auto-fit` to wrap cards gracefully.

**Typography (Strict Hierarchy)**:
- **Headings (H1-H6)**: Use `Anton` font, `font-weight: 400`, `text-transform: uppercase`, `letter-spacing: 0.02em`.
- **Base Body Text**: Use `Inter` font, `font-weight: 400`.
- **Labels, Subtext & Intros**: Use `Inter` font, `font-weight: 300` (Crucial for a premium, lightweight editorial feel).
- **Eyebrows/Micro-labels**: `font-size: 11px`, `font-weight: 700`, uppercase, `letter-spacing: 0.12em`.

**Architecture & Layout**:
- **Borders**: Structure the entire layout exclusively using `1px solid var(--line)` borders. 
- **Theming**: Implement a CSS Variable-based Light/Dark mode toggled via a `data-theme="dark"` attribute.
- **Panels & Cards**: Wrap content in flat structural cards using `background: var(--panel)`, `border-radius: 8px`, and `border: 1px solid var(--line)`. 
- **Footer Mobile Layout**: Footer links must ALWAYS remain in a horizontal row (`flex-direction: row`), even on mobile viewports. Do not stack them vertically.
- **Iconography**: Liberally use 36px squared icon boxes with a light background and 1px border to anchor list items and actions.

**Forms & Buttons (Strictly Pill-Shaped)**:
- **Inputs & Textareas**: MUST be fully rounded (pill-shaped). Use generous padding (e.g., `14px 24px`), `border: 1px solid var(--line)`, and `border-radius: 9999px`.
- **Action Buttons & CTAs**: ALL primary buttons, links, and form submits MUST be fully rounded/pill-shaped (`border-radius: 9999px`). The 8px border-radius is strictly reserved for structural panels, NEVER buttons. 
- **Focus States**: Remove default outline rings. Change the 1px border color to the primary brand accent color on focus.

## Modules

### Machine Learning Pipeline
- Data loading and pre-processing script.
- Random Forest model training on the 8 clinical parameters.
- Model evaluation and serialization (`model.pkl`).

### Public Interface
- Landing page with project context.
- Minimalist assessment form capturing 8 health metrics (Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age).

### Backend API (Flask)
- Request handler to process form submissions.
- Inference engine loading the `.pkl` model to generate mathematical risk predictions.
- LLM API integration to translate mathematical predictions and WHO guidelines into a natural-language clinical narrative.

### Results Dashboard
- Display of calculated risk probability.
- Display of the AI-generated natural-language explanation.
- Disclaimer noting the tool is for screening, not medical diagnosis.

## Data & Knowledge Base

- **Dataset:** Pima Indians Diabetes Dataset (.csv).
- **Hardcoded Logic:** WHO / IDF clinical thresholds for Glucose and BMI injected into the LLM prompt to ground explanations in medical facts.

## Security

- Input validation and sanitization for all 8 form fields.
- Environment variables (`.env`) for LLM API keys.
- Basic secure headers.

## Folder Structure

/model
/static
  /css
  /js
  /assets
/templates
/config

## Configuration

Create `.env` file with:
- FLASK_ENV
- FLASK_APP
- LLM_API_KEY

## Deliverables

- Complete Python source code
- Trained `.pkl` model
- Processed dataset `.csv`
- Assets (Fonts, Icons)
- README with setup instructions
- Final ZIP for execution

## Development Plan

1. Foundation & Dataset Sourcing
2. ML Model Training & Serialization
3. Flask Backend Integration
4. LLM Narrative Layer Setup
5. Frontend Form Implementation
6. Results Display
7. Final Packaging

## Attached Reference Documents

Use these as authoritative functional references:
- DIABETES RISK PREDICTION WEB APP AN AI.docx
- Pima_Indians_Diabetes_Dataset.csv