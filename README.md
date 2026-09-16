# DataPilot AI - Data Quality Agent

An AI-powered data quality agent built with Python, OpenAI Agents SDK, Gemini, Pandas, and public API data.

## Features

- Analyze CSV files
- Retrieve data from a public API
- Detect missing values
- Detect duplicate rows
- Analyze data types
- Analyze unique values
- Calculate numeric statistics
- Detect statistical outliers
- Validate common fields
- Clean CSV files when explicitly requested

## Architecture

User
↓
AI Data Quality Agent
↓
OpenAI Agents SDK
↓
Gemini
↓
Tools
├── CSV Quality Tool
├── Public API Tool
└── CSV Cleaning Tool
↓
Pandas
↓
Data Quality Report

## Technologies

- Python
- OpenAI Agents SDK
- Gemini API
- Pandas
- Requests
- FakeStoreAPI

## Setup

Install dependencies:

```bash
pip install -r requirements.txt