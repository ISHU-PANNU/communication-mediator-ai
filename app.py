from dotenv import load_dotenv
load_dotenv()  # reads .env file





# ── Imports ────────────────────────────────────────────────
import os, re, csv, torch
from datetime import datetime
from transformers import BertTokenizer, BertForSequenceClassification
from groq import Groq
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import gradio as gr
import nltk
nltk.download('punkt', quiet=True)

# ── Load BERT ───────────────────────────────────────────────
tokenizer  = BertTokenizer.from_pretrained("bert_style_classifier")
model_bert = BertForSequenceClassification.from_pretrained("bert_style_classifier")
model_bert.eval()
id2label   = {0: "anxious", 1: "avoidant", 2: "aggressive", 3: "healthy"}

# ── Groq ─────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client       = Groq(api_key=GROQ_API_KEY)
analyzer     = SentimentIntensityAnalyzer()

# ── Paste ALL your functions here ──────────────────────────
# ── Feedback Storage ────────────────────────────────────────
FEEDBACK_FILE = "feedback_data.csv"

def save_feedback(original, style, rewrite, rating, correction):
    file_exists = os.path.exists(FEEDBACK_FILE)
    with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "original_message", "detected_style",
                             "model_rewrite", "user_rating", "user_correction"])
        writer.writerow([
            datetime.now().isoformat(),
            original, style, rewrite, rating, correction
        ])
    total = sum(1 for _ in open(FEEDBACK_FILE)) - 1
    return f"✅ Feedback saved! ({total} total samples collected for training)"

def load_feedback_stats():
    if not os.path.exists(FEEDBACK_FILE):
        return "No feedback collected yet."
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return "No feedback collected yet."
    total   = len(rows)
    ratings = [r["user_rating"] for r in rows if r["user_rating"]]
    avg     = sum(int(r) for r in ratings) / len(ratings) if ratings else 0
    with_correction = sum(1 for r in rows if r["user_correction"].strip())
    return f"📊 {total} samples | ⭐ Avg rating: {avg:.1f}/5 | ✏️ {with_correction} corrections provided"

# ── Last result state for feedback ─────────────────────────
last_result = {"original": "", "style": "", "rewrite": ""}

# ── 1. BERT Style Classifier ────────────────────────────────
def classify_style(text):
    text_lower = text.lower()

    # Rule-based override for clear anxious signals
    anxious_signals = [
        "you are avoiding", "you ignore", "am i nothing",
        "do you care", "you never reply", "don't leave",
        "please talk", "can we talk", "i miss you",
        "are you there", "why don't you", "i need you"
    ]
    for signal in anxious_signals:
        if signal in text_lower:
            return "anxious", 95.0

    # BERT classification
    inputs = tokenizer(
        text, return_tensors="pt",
        truncation=True, padding=True, max_length=128
    )
    with torch.no_grad():
        outputs = model_bert(**inputs)
    probs      = torch.softmax(outputs.logits, dim=1)[0]
    pred_id    = torch.argmax(probs).item()
    confidence = round(probs[pred_id].item() * 100, 1)
    return id2label[pred_id], confidence

# ── 2. Sentiment Analysis ───────────────────────────────────
def get_sentiment(text):
    score = analyzer.polarity_scores(str(text))
    if score["compound"] >= 0.05:
        return "positive"
    elif score["compound"] <= -0.05:
        return "negative"
    else:
        return "neutral"

# ── 3. Speech Act Detector ──────────────────────────────────
def detect_speech_act(text):
    text_lower = text.lower().strip()
    if re.search(r'\b(you always|you never|you made|your fault|because of you|you ruined)\b', text_lower):
        return "accusation"
    elif re.search(r'\b(fine|whatever|forget it|never mind|i don.t care|doesn.t matter|leave me alone)\b', text_lower):
        return "withdrawal"
    elif re.search(r'\b(please|i need you|don.t leave|stay|come back|i beg|am i nothing|do i matter|do you even)\b', text_lower):
        return "plea"
    elif text.strip().endswith('?') or re.search(r'\b(what|why|how|when|where|who|can we|do you|are you)\b', text_lower):
        return "question"
    elif re.search(r'\b(i feel|i am feeling|i felt|i.m feeling|it hurts|i.m sad|i.m scared|i miss)\b', text_lower):
        return "expression"
    elif re.search(r'\b(i think|i believe|i know|i want|i need|in my opinion)\b', text_lower):
        return "assertion"
    elif re.search(r'\b(or else|you.ll regret|i.m done|i.m leaving|this is over|i quit)\b', text_lower):
        return "threat"
    else:
        return "statement"

# ── 4. Groq Rewriter ────────────────────────────────────────
def rewrite_message(text, style, speech_act, sentiment, context):
    prompt = f"""You are a communication coach trained in attachment theory.

Style definitions:
- Anxious: fear of abandonment, seeks reassurance ("you never care", "I knew you'd leave")
- Avoidant: shuts down, withdraws ("fine", "whatever", "I don't care")
- Aggressive: blaming, attacking ("you ALWAYS", "unacceptable", "your fault")
- Healthy: calm, clear, uses "I feel" statements, solution-focused

Message: "{text}"
Detected Style: {style}
Speech Act: {speech_act}
Sentiment: {sentiment}
Context: {context}

Rewriting Rules:
- Write like a real person texting a friend, NOT like a therapist or HR email
- Use simple everyday words — no "I'd love to", "certainly", "I appreciate"
- Keep same energy and length as original message
- Short message = short rewrite (1 sentence max)
- If personal context: casual, warm, direct
- If professional context: polite but still human, not robotic

Bad example:  "I'd love to talk it through with you"
Good example: "hey can we just talk? i really need you right now"

Format:
Rewrite: <rewritten message>
Why: <one sentence>"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a communication coach who writes like a real human, not a therapist. Keep it casual and natural."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.6,
        max_tokens=150
    )
    return response.choices[0].message.content

# ── 5. Full Pipeline ─────────────────────────────────────────
def full_pipeline(text, context):
    if not text.strip():
        return "Please enter a message.", ""

    style, confidence = classify_style(text)
    sentiment         = get_sentiment(text)
    speech_act        = detect_speech_act(text)
    rewrite           = rewrite_message(text, style, speech_act, sentiment, context)

    style_emoji = {
        "anxious":    "😰",
        "avoidant":   "🚪",
        "aggressive": "😠",
        "healthy":    "✅"
    }
    sentiment_emoji = {
        "positive": "😊",
        "negative": "😞",
        "neutral":  "😐"
    }

    analysis = f"""{style_emoji.get(style, '')} Style      : {style.upper()} ({confidence}% confidence)
{sentiment_emoji.get(sentiment, '')} Sentiment  : {sentiment.upper()}
💬 Speech Act : {speech_act.upper()}"""

    return analysis, rewrite

# ── 6. Pipeline wrapper for UI (stores result for feedback) ─
def run_pipeline(text, context):
    if not text.strip():
        return "Please enter a message.", "", gr.update(visible=False)

    analysis, rewrite = full_pipeline(text, context)

    # Store for feedback
    style_line = [l for l in analysis.split("\n") if "Style" in l]
    style = style_line[0].split(":")[1].split("(")[0].strip().lower() if style_line else "unknown"
    last_result["original"] = text
    last_result["style"]    = style
    last_result["rewrite"]  = rewrite

    return analysis, rewrite, gr.update(visible=True)

# ── 7. Feedback submission ───────────────────────────────────
def submit_feedback(rating, correction):
    if not last_result["original"]:
        return "⚠️ Please analyze a message first before submitting feedback."
    return save_feedback(
        last_result["original"],
        last_result["style"],
        last_result["rewrite"],
        rating,
        correction or ""
    )
```

---

## Complete `app.py` Order

Make sure your file is structured like this:
```
1. Imports
2. Load BERT
3. Groq client
4. ← Paste above functions here
5. Paste Gradio UI (with gr.Blocks)
6. app.launch()

# ── Paste your full Gradio UI here ─────────────────────────
import gradio as gr
import json
import os
import csv
from datetime import datetime

# ── Feedback Storage ────────────────────────────────────────
FEEDBACK_FILE = "feedback_data.csv"

def save_feedback(original, style, rewrite, rating, correction):
    """Save user feedback to CSV for future model training."""
    file_exists = os.path.exists(FEEDBACK_FILE)
    with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "original_message", "detected_style",
                             "model_rewrite", "user_rating", "user_correction"])
        writer.writerow([
            datetime.now().isoformat(),
            original, style, rewrite, rating, correction
        ])
    total = sum(1 for _ in open(FEEDBACK_FILE)) - 1
    return f"✅ Feedback saved! ({total} total samples collected for training)"

def load_feedback_stats():
    """Load feedback statistics."""
    if not os.path.exists(FEEDBACK_FILE):
        return "No feedback collected yet."
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return "No feedback collected yet."
    total    = len(rows)
    ratings  = [r["user_rating"] for r in rows if r["user_rating"]]
    avg      = sum(int(r) for r in ratings) / len(ratings) if ratings else 0
    with_correction = sum(1 for r in rows if r["user_correction"].strip())
    return f"📊 {total} samples | ⭐ Avg rating: {avg:.1f}/5 | ✏️ {with_correction} corrections provided"

# ── CSS ─────────────────────────────────────────────────────
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

:root {
    --bg-deep:    #0a0e1a;
    --bg-card:    #111827;
    --bg-input:   #1a2235;
    --accent:     #6ee7b7;
    --accent2:    #38bdf8;
    --accent3:    #f472b6;
    --text-main:  #e2e8f0;
    --text-muted: #64748b;
    --border:     #1e293b;
    --radius:     14px;
}

* { box-sizing: border-box; }

body, .gradio-container {
    background: var(--bg-deep) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text-main) !important;
}

/* ── Hero ── */
.hero-section {
    text-align: center;
    padding: 48px 24px 32px;
    position: relative;
    overflow: hidden;
}

.hero-section::before {
    content: '';
    position: absolute;
    top: -80px; left: 50%;
    transform: translateX(-50%);
    width: 600px; height: 300px;
    background: radial-gradient(ellipse, rgba(110,231,183,0.12) 0%, transparent 70%);
    pointer-events: none;
}

.hero-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.6rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #6ee7b7 0%, #38bdf8 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 12px !important;
    letter-spacing: -1px;
    animation: fadeSlideDown 0.8s ease both;
}

.hero-sub {
    font-size: 1rem !important;
    color: var(--text-muted) !important;
    font-weight: 300 !important;
    animation: fadeSlideDown 0.8s ease 0.15s both;
}

.badge-row {
    display: flex;
    gap: 10px;
    justify-content: center;
    flex-wrap: wrap;
    margin-top: 16px;
    animation: fadeSlideDown 0.8s ease 0.3s both;
}

.badge {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 0.75rem;
    color: var(--text-muted);
    letter-spacing: 0.5px;
}

/* ── Style Cards ── */
.style-cards {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    padding: 0 16px 32px;
    animation: fadeSlideUp 0.7s ease 0.4s both;
}

.style-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s;
}
.style-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
}
.style-card.anxious  { border-top: 3px solid #f472b6; }
.style-card.avoidant { border-top: 3px solid #38bdf8; }
.style-card.aggressive { border-top: 3px solid #fb923c; }
.style-card.healthy  { border-top: 3px solid #6ee7b7; }

.style-emoji { font-size: 1.8rem; margin-bottom: 6px; }
.style-name  { font-family: 'Syne', sans-serif; font-size: 0.85rem; font-weight: 700; color: var(--text-main); }
.style-desc  { font-size: 0.72rem; color: var(--text-muted); margin-top: 4px; line-height: 1.4; }

/* ── Main Panel ── */
.main-panel {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    padding: 0 16px 24px;
}

.panel-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
}

.panel-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 12px;
}

/* ── Gradio overrides ── */
.gradio-container label {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
}

textarea, input[type="text"] {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-main) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s !important;
}

textarea:focus, input:focus {
    border-color: var(--accent) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(110,231,183,0.1) !important;
}

/* ── Buttons ── */
button.primary {
    background: linear-gradient(135deg, #6ee7b7, #38bdf8) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #0a0e1a !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.5px !important;
    padding: 12px 24px !important;
    transition: opacity 0.2s, transform 0.15s !important;
}
button.primary:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}

button.secondary {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-muted) !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    transition: border-color 0.2s !important;
}
button.secondary:hover { border-color: var(--accent) !important; }

/* ── Analysis output ── */
.analysis-box {
    background: var(--bg-input) !important;
    border-left: 3px solid var(--accent) !important;
    border-radius: 10px !important;
    padding: 16px !important;
    font-family: 'DM Sans', monospace !important;
    font-size: 0.9rem !important;
    line-height: 1.8 !important;
}

/* ── Feedback section ── */
.feedback-section {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-top: 3px solid var(--accent3);
    border-radius: var(--radius);
    padding: 24px;
    margin: 0 16px 24px;
    animation: fadeSlideUp 0.7s ease 0.5s both;
}

/* ── Examples ── */
.examples-section {
    padding: 0 16px 24px;
}

.example-tag {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 0.82rem;
    color: var(--text-muted);
    cursor: pointer;
    transition: all 0.2s;
    display: inline-block;
    margin: 4px;
}
.example-tag:hover {
    border-color: var(--accent);
    color: var(--accent);
}

/* ── Stats bar ── */
.stats-bar {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 20px;
    margin: 0 16px 16px;
    font-size: 0.82rem;
    color: var(--text-muted);
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Animations ── */
@keyframes fadeSlideDown {
    from { opacity: 0; transform: translateY(-16px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.5; }
}
.analyzing { animation: pulse 1.2s ease infinite; }

/* ── Radio buttons ── */
.gradio-radio label {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
}
.gradio-radio input:checked + label {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* ── Accordion ── */
.accordion {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 24px;
    color: var(--text-muted);
    font-size: 0.78rem;
    border-top: 1px solid var(--border);
    margin-top: 8px;
}
"""

# ── State for feedback ───────────────────────────────────────
last_result = {"original": "", "style": "", "rewrite": ""}

def run_pipeline(text, context):
    """Wrapper that also stores result for feedback."""
    if not text.strip():
        return "Please enter a message.", "", gr.update(visible=False)

    analysis, rewrite = full_pipeline(text, context)

    # Store for feedback
    style_line = [l for l in analysis.split("\n") if "Style" in l]
    style = style_line[0].split(":")[1].split("(")[0].strip().lower() if style_line else "unknown"
    last_result["original"] = text
    last_result["style"]    = style
    last_result["rewrite"]  = rewrite

    return analysis, rewrite, gr.update(visible=True)

def submit_feedback(rating, correction):
    """Save feedback and return confirmation."""
    if not last_result["original"]:
        return "⚠️ Please analyze a message first before submitting feedback."
    msg = save_feedback(
        last_result["original"],
        last_result["style"],
        last_result["rewrite"],
        rating,
        correction or ""
    )
    return msg

# ── Build UI ─────────────────────────────────────────────────
with gr.Blocks(
    title="Communication Mediator AI",
    css=custom_css,
    theme=gr.themes.Base()
) as app:

    # ── Hero ──
    gr.HTML("""
    <div class="hero-section">
        <div class="hero-title">Communication Mediator AI</div>
        <div class="hero-sub">Detects unhealthy communication patterns & rewrites them with empathy</div>
        <div class="badge-row">
            <span class="badge">BERT Classifier</span>
            <span class="badge">LLaMA 3 Rewriter</span>
            <span class="badge">VADER Sentiment</span>
            <span class="badge">Speech Act Detection</span>
            <span class="badge">NLP · IGDTUW CSE</span>
        </div>
    </div>
    """)

    # ── Style Cards ──
    gr.HTML("""
    <div class="style-cards">
        <div class="style-card anxious">
            <div class="style-emoji">😰</div>
            <div class="style-name">Anxious</div>
            <div class="style-desc">Fear of abandonment, seeks constant reassurance</div>
        </div>
        <div class="style-card avoidant">
            <div class="style-emoji">🚪</div>
            <div class="style-name">Avoidant</div>
            <div class="style-desc">Shuts down, withdraws, dismissive</div>
        </div>
        <div class="style-card aggressive">
            <div class="style-emoji">😠</div>
            <div class="style-name">Aggressive</div>
            <div class="style-desc">Blaming, attacking, accusatory tone</div>
        </div>
        <div class="style-card healthy">
            <div class="style-emoji">✅</div>
            <div class="style-name">Healthy</div>
            <div class="style-desc">Calm, clear "I feel" statements</div>
        </div>
    </div>
    """)

    # ── Main Input / Output ──
    with gr.Row(equal_height=True):
        with gr.Column():
            input_text = gr.Textbox(
                label="Your Message",
                placeholder="Type the message you want to analyze and rewrite...",
                lines=5
            )
            context = gr.Radio(
                choices=["personal", "professional"],
                value="personal",
                label="Context"
            )
            with gr.Row():
                submit_btn = gr.Button("✦ Analyze & Rewrite", variant="primary", scale=3)
                clear_btn  = gr.Button("Clear", variant="secondary", scale=1)

        with gr.Column():
            analysis_out = gr.Textbox(
                label="Analysis",
                lines=4,
                interactive=False,
                elem_classes=["analysis-box"]
            )
            rewrite_out = gr.Textbox(
                label="Rewritten Message",
                lines=5,
                interactive=False
            )

    # ── Feedback Section ──
    with gr.Group(visible=False) as feedback_group:
        gr.HTML("<div style='padding: 8px 0 4px; font-family: Syne; font-size: 0.75rem; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; color: #64748b;'>✦ Help Improve the Model</div>")

        with gr.Row():
            rating = gr.Slider(
                minimum=1, maximum=5, step=1, value=3,
                label="Rate the Rewrite (1 = Poor, 5 = Excellent)"
            )
            correction = gr.Textbox(
                label="Your Better Version (optional)",
                placeholder="If the rewrite wasn't good, type a better version here...",
                lines=2
            )

        with gr.Row():
            feedback_btn = gr.Button("Submit Feedback 💬", variant="primary")
            feedback_out = gr.Textbox(label="", interactive=False, lines=1)

    # ── Examples ──
    gr.HTML("<div style='padding: 16px 0 8px 4px; font-family: Syne; font-size: 0.75rem; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; color: #64748b;'>✦ Try These Examples</div>")

    gr.Examples(
        examples=[
            ["You never reply to me. Do you even care?", "personal"],
            ["Fine. Whatever. I don't even care anymore.", "personal"],
            ["am i nothing for you", "personal"],
            ["You always miss deadlines. This is completely unacceptable.", "professional"],
            ["I'm done trying. You clearly don't care.", "personal"],
            ["can we talk? i need you", "personal"],
            ["Your work is always below standard.", "professional"],
            ["I feel disconnected lately. Can we talk?", "personal"],
        ],
        inputs=[input_text, context],
        label=""
    )

    # ── Feedback Stats ──
    with gr.Accordion("📊 Training Data Stats", open=False):
        stats_out = gr.Textbox(
            value=load_feedback_stats(),
            interactive=False,
            label=""
        )
        refresh_btn = gr.Button("Refresh Stats", variant="secondary")

    # ── Footer ──
    gr.HTML("""
    <div class="footer">
        Communication Mediator AI · NLP Project · IGDTUW CSE (AI & DS) ·
        Built with BERT + LLaMA3 + Gradio
    </div>
    """)

    # ── Actions ──
    submit_btn.click(
        fn=run_pipeline,
        inputs=[input_text, context],
        outputs=[analysis_out, rewrite_out, feedback_group]
    )

    clear_btn.click(
        fn=lambda: ("", "", "", gr.update(visible=False)),
        inputs=[],
        outputs=[input_text, analysis_out, rewrite_out, feedback_group]
    )

    feedback_btn.click(
        fn=submit_feedback,
        inputs=[rating, correction],
        outputs=[feedback_out]
    )

    refresh_btn.click(
        fn=load_feedback_stats,
        inputs=[],
        outputs=[stats_out]
    )

app.launch(share=True)

app.launch()   # No share=True needed
