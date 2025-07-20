# AI-Powered GPO Customer Matcher

An AI-driven customer record matcher designed for **Group Purchasing Organizations (GPOs)**. Combines **fuzzy logic** and **Groq LLM (Llama 3 / Mixtral)** to deliver highly accurate entity matching across customer data sources.

---

## Features

- ✅ 95%+ accuracy in customer matching  
- 🔍 Fuzzy logic for efficient pre-filtering  
- 🤖 LLM support using Groq API (Llama 3 8B/70B, Mixtral 8x7B)  
- 📊 Dynamic visual analytics (confidence, quality distribution)  
- 📁 Multi-file CSV upload and pairwise matching  
- 🎛️ Configurable AI confidence and fuzzy thresholds  
- 📤 Export options: CSV and JSON  
- 🔐 Secure Groq API key input via Streamlit sidebar  

---

##  Architecture Flow

1. Upload CSV files (minimum 2)  
2. Fuzzy matching filters high-probability pairs  
3. Groq LLM evaluates and scores precision matches  
4. Match decision with reasoning and confidence  
5. Final output with match quality categorization  
6. Visual analytics and exportable results  

---

## Tech Stack

- **Frontend**: Streamlit  
- **Matching Engine**: RapidFuzz, Groq LLM  
- **Visualization**: Plotly  
- **Language**: Python 3.9+  
- **Deployment**: Local machine / Streamlit Cloud  

---

