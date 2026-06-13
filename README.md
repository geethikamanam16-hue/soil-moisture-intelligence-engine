# Soil Moisture Intelligence Engine

The **Soil Moisture Intelligence Engine** is a professional-grade, multi-modal Earth science analysis platform. It integrates local Zarr-based historical dataset processing, Google Earth Engine (GEE) cloud satellite cross-validation, and local multi-modal Retrieval-Augmented Generation (RAG) to analyze, visualize, and interpret soil moisture dynamics.

---

## Core Features

### 1. Local Zarr Dataset Analysis
- **Historical Scope**: Processes multi-gigabyte historical AMSR2-based soil moisture datasets spanning from 2002 to 2023.
- **Smart Temporal Parsing**: Features advanced date parsing algorithms that automatically resolve standard ISO dates, specific quarters (Q1–Q4), and meteorological seasons (e.g., "Winter 2015", "Summer 2020") into coordinate slices.

### 2. Multi-Modal Scientific RAG
- **Vision-Language Model**: Leverages a local `llava:7b` instance to read, interpret, and explain scientific tables, charts, and figures.
- **Dynamic Cropping**: Directly extracts and analyzes targeted segments and figures from PDF literature assets.
- **Ambiguity Detection**: Evaluates query context and figure clarity, automatically flagging potential ambiguities or incomplete data within the visual assets to prevent hallucinated insights.

### 3. Google Earth Engine (GEE) Integration
- **SMAP Cross-Validation**: Incorporates a Cloud SMAP tab to fetch and analyze multi-year satellite validation datasets.
- **Dynamic GCP Project ID Injection**: Supports dynamic Google Cloud Project ID injection for streamlined, authenticated access to the Earth Engine API.

### 4. HD Dynamic Visualizations
- **High-Resolution Graphics**: Generates 300-DPI publication-ready multi-panel spatial and temporal maps.
- **Interactive UI Integration**: Renders spatial anomalies, trend slopes, and mean distribution maps natively within the dashboard.

### 5. Intelligent Topic Guardrails
- **Semantic Classification**: Employs robust intent classifiers to safely route Earth science, hydrology, and remote sensing queries.
- **Off-Topic Rejector**: Automatically intercepts and rejects irrelevant prompts to safeguard compute resources and maintain professional focus.

---

## Architecture Overview

- **`app.py`**: Streamlit-based web dashboard coordinating the visualization tabs, document QA, and dataset controls.
- **`engine.py`**: Core local analytics engine managing Zarr chunk reading, statistical aggregations (means, trends), and temporal slicing.
- **`gee_smap.py`**: Handler for Google Earth Engine API calls, cloud dataset queries, and SMAP data retrieval.
- **`vision_q.py` & `literature_qa.py`**: Orchestrates PDF parsing, image cropping, and prompt management for the local `llava:7b` vision LLM.
- **`Query_classifier.py` & `intent_classifier.py`**: Manages semantic guardrails and classification of user inputs.
- **`Config.py`**: Centralized configuration management.

---

## Setup & Installation

Follow these steps to configure and run the engine locally:

### 1. Clone & Prepare Environment
Ensure you have Python 3.10+ installed. Navigate to the project directory and install the required Python packages:
```bash
pip install -r requirements.txt
```

### 2. Configure Local Vision Model
Download and start Ollama, then pull the `llava:7b` model to enable multi-modal RAG capabilities:
```bash
ollama pull llava:7b
```

### 3. Authenticate Google Earth Engine
Ensure you have a Google Cloud Project with Earth Engine API enabled. Authenticate your local machine using the command line:
```bash
earthengine authenticate
```

### 4. Run the Application
Launch the Streamlit web dashboard:
```bash
streamlit run app.py
```
