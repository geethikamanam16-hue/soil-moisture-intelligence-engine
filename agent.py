"""
Agent - Ollama LLM Interface for Query Processing

Converts natural language to structured parameters with fallback support.

Used as a refinement layer on top of the rule-based QueryClassifier.
"""

import requests
import json


class OllamaAgent:
    """Interfaces with Ollama to refine unclear query classifications."""

    def __init__(self, model_name="qwen2.5:3b"):
        import Config
        self.url     = Config.OLLAMA_URL
        self.model   = model_name
        self.timeout = Config.OLLAMA_TIMEOUT

    def process_query(self, classifier_output: dict) -> dict:
        """
        Optionally refine classifier output using Ollama for ambiguous queries.

        Parameters
        ----------
        classifier_output : dict from QueryClassifier.classify()

        Returns
        -------
        dict with same structure, possibly refined
        """
        # Only call Ollama if the query was unclear/ambiguous
        if classifier_output.get('query_clarity') in ['unclear', 'ambiguous']:
            refined = self._refine_with_ollama(classifier_output)
            if refined:
                return refined

        # Pass through classifier output as-is
        return classifier_output

    def _refine_with_ollama(self, classifier_output: dict):
        """
        Ask Ollama to confirm or correct ambiguous classifier output.

        Returns refined dict on success, None on failure.
        """
        prompt = f"""
A soil moisture analysis query was classified with some uncertainty.

Classifier interpretation:
  - Region:      {classifier_output.get('region', 'unknown')}
  - Operation:   {classifier_output.get('operation', 'unknown')}
  - Output type: {classifier_output.get('output_type', 'both')}
  - Start date:  {classifier_output.get('start_date', 'unknown')}
  - End date:    {classifier_output.get('end_date', 'unknown')}
  - Clarity:     {classifier_output.get('query_clarity', 'unclear')}
  - Comparison type: {classifier_output.get('comparison_type', 'N/A')}
  - Comparison metric: {classifier_output.get('comparison_metric', 'N/A')}

Valid operations: mean, slope, minimum, maximum, comparison
Valid output types: scalar, map, both
Valid comparison types: time (same region, two periods), region (two regions)
Valid comparison metrics: mean, min, max

Please respond ONLY with a JSON object (no markdown, no extra text) with:
{{
  "region": "...",
  "operation": "...",
  "output_type": "...",
  "comparison_type": "...",
  "comparison_metric": "...",
  "confident": true or false
}}

Only set "confident": true if you are sure of the corrections.
"""
        try:
            response = requests.post(
                self.url,
                json={
                    "model":  self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options":{
                        "temperature":0.1,
                        "top_p":0.2
                    }
                },
                timeout=self.timeout
            )

            response.raise_for_status()

            raw_text = response.json().get('response', '{}')
            raw_text = raw_text.replace("```json", "").replace("```", "").strip()

            parsed = json.loads(raw_text)

            if parsed.get('confident'):
                refined = dict(classifier_output)   # copy

                if 'region' in parsed:
                    refined['region'] = parsed['region']

                if 'operation' in parsed:
                    refined['operation'] = parsed['operation']

                if 'output_type' in parsed:
                    refined['output_type'] = parsed['output_type']

                if 'comparison_type' in parsed:
                    refined['comparison_type'] = parsed['comparison_type']

                if 'comparison_metric' in parsed:
                    refined['comparison_metric'] = parsed['comparison_metric']

                refined['source'] = 'ollama_refined'

                return refined

        except requests.exceptions.Timeout:
            print("⚠️  Ollama timeout — using classifier result directly.")

        except Exception:
            print("⚠️  Ollama unavailable — using classifier result directly.")

        return None