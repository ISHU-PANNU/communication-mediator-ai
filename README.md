---
title: Communication Mediator AI
emoji: 🧠
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: true
---

# 🧠 Communication Mediator AI

> Detects anxious, avoidant & aggressive communication patterns  
> and rewrites them into healthy communication using NLP + LLM.

## 🔗 Live Demo
[Try it on HuggingFace Spaces]
git clone https://huggingface.co/spaces/ISHU08/communication-mediator-ai

##  How It Works
User Message
     ↓
BERT Classifier → detects style (anxious/avoidant/aggressive/healthy)
     ↓
VADER → sentiment analysis (positive/negative/neutral)
     ↓
Rule-based → speech act detection (accusation/plea/withdrawal...)
     ↓
LLaMA 3 → rewrites into healthy communication
     ↓
Gradio UI → shows analysis + rewritten message


## 🛠️ Built With
- BERT (fine-tuned on GoEmotions — 40K samples)
- LLaMA 3 via Groq API
- VADER Sentiment Analysis
- Rule-based Speech Act Detection (Pragmatics)
- Gradio UI with feedback system

## 📦 Datasets Used
| Dataset             |Use                       |
|---------------------|--------------------------|
| GoEmotions (Google) | BERT classifier training |
| EmphaticDialogues   | Rewriter context         |
| TweetEval           | Sentiment analysis       |

## 🎯 Use Cases
-  Personal relationships (anxious/avoidant)
-  HR & workplace conflict resolution
-  Mental health & therapy support
-  Pre-send message checker



## 📁 Project Structure

├── app.py                    # Main application
├── requirements.txt          # Dependencies
├── bert_style_classifier/    # Fine-tuned BERT model
└── feedback_data.csv         # Auto-generated training feedback


## 📊 Model Performance
- BERT Classifier:
                     F1 score  
   anxious            0.70       
   avoidant           0.83      
   aggressive         0.64       
   healthy            0.61 
- Rewriter BLEU Score: 0.0307

