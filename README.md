
# Autonomous Supply Chain Resilience Agent

An AI-powered Supply Chain Resilience Agent built using **Google ADK + Gemini**.

This agent acts as an intelligent operations co-pilot for mid-market manufacturers by detecting disruption signals, assessing operational risk, and recommending mitigation strategies.

---

## Problem

Modern supply chains face increasing disruption risks:

- Supplier shutdowns
- Port congestion
- Climate-related delays
- Geopolitical instability

Mid-market manufacturers often lack structured tools to proactively assess risk and simulate mitigation strategies.

---

## Solution

This project demonstrates an autonomous agent that:

- Monitors disruption signals (mock data)
- Classifies disruption type and severity
- Estimates disruption probability
- Calculates revenue-at-risk
- Simulates mitigation trade-offs
- Drafts supplier outreach emails
- Logs disruption cases into memory

The system follows a structured pipeline:

Perception → Risk Assessment → Planning → Action → Memory

---

## 🏗 Project Structure

```
agent/
  agent.py        # ADK agent definition
  tools.py        # Risk engine, simulations, email drafting

data/
  company_profile.json
  disruptions.json
  memory.json
```

---

##  How To Run

### 1. Clone the repository

```bash
git clone https://github.com/amelia-haa/supply-chain-agent.git
cd supply-chain-agent
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create a `.env` file

In the project root, create a file named `.env` and add:

```
GOOGLE_GENAI_USE_VERTEXAI=0
GOOGLE_API_KEY=YOUR_GOOGLE_AI_STUDIO_API_KEY
```

Generate your API key here:
https://aistudio.google.com/apikey

### 5. Run the agent

```bash
python -m google.adk web
```

Open your browser at:

http://127.0.0.1:8000

---

## 🔒 Security

API keys are NOT included in this repository.
Each user must provide their own API key locally in a `.env` file.

---

## 🏆 Built For

Autonomous Supply Chain Resilience Hackathon  
Google ADK Workshop 2026
