---
title: Intelligent Feedback Analysis System
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---

# Intelligent User Feedback Analysis and Action System

A **multi-agent AI system** built with CrewAI + Google Gemini that automatically processes user feedback from app stores and support emails, classifies it, and generates structured engineering tickets.

## Agents

| Agent | Role |
|-------|------|
| CSV Reader Agent | Parses and structures raw feedback |
| Feedback Classifier Agent | Classifies into Bug / Feature Request / Praise / Complaint / Spam |
| Bug Analysis Agent | Extracts severity, device info, steps to reproduce |
| Feature Extractor Agent | Identifies features and estimates user impact |
| Ticket Creator Agent | Generates structured tickets |
| Quality Critic Agent | Reviews and improves ticket quality |

## Setup

1. Enter your **Google API Key** in the sidebar
2. Adjust processing settings (batch size, confidence threshold, priority overrides)
3. Click **Start Pipeline** to run the full multi-agent workflow
4. View, filter, edit, and download generated tickets

## Environment Variables (for HF Spaces secrets)

- `GOOGLE_API_KEY` — Your Google Gemini API key
