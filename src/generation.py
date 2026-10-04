"""
generation.py
=============
LLM generation module for the cybersecurity RAG system.

Design decisions:
  - LLM: Mistral-7B-Instruct-v0.2 (open-source, Apache 2.0 licence).
    Accessed via Ollama local inference server (CPU-safe, no GPU needed)
    OR via HuggingFace transformers pipeline as fallback.
    Chosen over GPT-4 because the assignment requires an open-source LLM.
  - Prompt design follows retrieved-context prompting best practices:
      1. System role establishes the cybersecurity expert persona.
      2. Context block contains top-3 retrieved passages with source labels.
      3. Instruction specifies output format: exactly 3 numbered guidelines.
      4. Query is separated from context for clarity.
  - Temperature=0.3 for reproducibility; top_p=0.9 for diversity.
  - Max new tokens=512 to keep outputs concise and avoid hallucinated padding.

Fallback chain:
  Ollama (Mistral 7B local) → HuggingFace pipeline → Mock generator (demo).
"""

import json
import re
import time
from typing import List, Dict, Optional
import requests


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior cybersecurity consultant with deep expertise in 
incident response, vulnerability management, phishing prevention, and secure cloud 
configuration. You advise mid-sized enterprise security teams.

Your task: given retrieved knowledge-base passages and a practitioner's question, 
generate EXACTLY 3 concise, actionable guidelines. Each guideline must:
  - Be grounded in the provided context (no fabrication).
  - Reference the specific control, standard, or tool mentioned in the context.
  - Be written for a technical security practitioner audience.
  - Be numbered 1, 2, 3 and start on a new line.

Do not include any preamble, caveats, or closing remarks. Output ONLY the 3 numbered guidelines."""


def build_prompt(query: str, context: str) -> str:
    """
    Construct the full prompt for the LLM.

    The format uses Mistral's [INST]...[/INST] chat template.
    For models served via Ollama, the system prompt is passed as a separate field.
    """
    return (
        f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
        f"RETRIEVED CONTEXT:\n{context}\n\n"
        f"PRACTITIONER QUERY: {query}\n\n"
        f"Generate 3 actionable cybersecurity guidelines based only on the context above. [/INST]"
    )


def build_openai_messages(query: str, context: str) -> List[Dict]:
    """
    Alternative message format for OpenAI-compatible endpoints (Ollama, LM Studio).
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"RETRIEVED CONTEXT:\n{context}\n\n"
                f"PRACTITIONER QUERY: {query}\n\n"
                f"Generate 3 actionable cybersecurity guidelines based only on the context above."
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Ollama client (primary)
# ---------------------------------------------------------------------------

class OllamaGenerator:
    """
    Calls a locally running Ollama server.
    Install: https://ollama.ai  |  ollama pull mistral
    """

    def __init__(
        self,
        model: str = "mistral",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.3,
        max_tokens: int = 512,
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens

    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def generate(self, query: str, context: str) -> Dict:
        """
        Generate guidelines. Returns dict with 'output', 'model', 'latency_s'.
        """
        messages = build_openai_messages(query, context)
        payload = {
            "model": self.model,
            "messages": messages,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "top_p": 0.9,
            },
            "stream": False,
        }
        t0 = time.time()
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        latency = round(time.time() - t0, 2)
        data = response.json()
        output_text = data["message"]["content"].strip()
        return {
            "output":    output_text,
            "model":     self.model,
            "latency_s": latency,
            "tokens":    data.get("eval_count", -1),
        }


# ---------------------------------------------------------------------------
# HuggingFace transformers fallback
# ---------------------------------------------------------------------------

class HFGenerator:
    """
    Uses HuggingFace transformers pipeline for local inference.
    Slower than Ollama but no separate server required.
    Suitable for CPU-only environments.
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        temperature: float = 0.3,
        max_new_tokens: int = 512,
    ):
        from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
        import torch
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.model_name = model_name
        print(f"[HFGenerator] Loading '{model_name}' (this may take a few minutes) …")
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        print("[HFGenerator] Ready.")

    def generate(self, query: str, context: str) -> Dict:
        prompt = build_prompt(query, context)
        t0 = time.time()
        result = self.pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=0.9,
            do_sample=True,
            return_full_text=False,
        )
        latency = round(time.time() - t0, 2)
        output_text = result[0]["generated_text"].strip()
        return {
            "output":    output_text,
            "model":     self.model_name,
            "latency_s": latency,
            "tokens":    -1,
        }


# ---------------------------------------------------------------------------
# Mock generator — for demonstration when no LLM server is available
# ---------------------------------------------------------------------------

MOCK_GUIDELINES = {
    "phishing": [
        "1. Deploy DMARC, DKIM, and SPF email authentication on all domains and set SPF to '-all' hard-fail mode to prevent domain spoofing, as recommended by NIST SP 800-177 and CISA guidance.",
        "2. Implement phishing-resistant MFA (FIDO2/passkeys) on all externally facing services and conduct quarterly simulated phishing campaigns with targeted remedial training for repeat clickers.",
        "3. Upon confirmed phishing incident, immediately revoke compromised session tokens, quarantine matching emails via the email purge API, and add attacker IOCs (URLs, sender domains, attachment hashes) to the blocklist within 30 minutes.",
    ],
    "incident_response": [
        "1. Follow the NIST SP 800-61r2 lifecycle — triage events by P1–P4 severity within defined SLA windows using SIEM correlation rules and EDR telemetry.",
        "2. For ransomware containment, isolate affected hosts via VLAN quarantine, snapshot volumes before remediation, and establish out-of-band communication if primary channels are compromised.",
        "3. Conduct a Post-Incident Review (PIR) within 5 business days, map the attacker kill-chain, report MTTD/MTTR metrics to leadership, and retain evidence for ≥3 years per ISO 27035.",
    ],
    "vulnerability_management": [
        "1. Prioritise CISA KEV catalogue entries above CVSS score as they indicate active exploitation; remediate KEV items within 14 days regardless of their base CVSS rating.",
        "2. Integrate Software Composition Analysis (SCA) tools (Snyk, Dependabot, or OWASP Dependency-Check) into CI/CD pipelines to automatically detect vulnerable third-party libraries.",
        "3. Conduct annual external penetration tests and ad-hoc tests after significant infrastructure changes; document findings in a remediation register and re-test to verify closure per NIST SP 800-115.",
    ],
    "cloud_security": [
        "1. Enable AWS CloudTrail logging in all regions with log-file integrity validation, and enforce S3 Block Public Access at the account level per CIS Benchmarks for AWS.",
        "2. Apply zero-trust micro-segmentation via service mesh (Istio/Linkerd) to authenticate every service-to-service API call with mTLS and short-expiry JWTs, minimising lateral movement blast radius.",
        "3. Remove root IAM access keys, enforce MFA on the root account, and use IAM roles for all EC2 workloads; enable AWS GuardDuty and Security Hub for continuous threat detection.",
    ],
    "default": [
        "1. Assess your current security posture against the NIST Cybersecurity Framework (CSF) five functions — Identify, Protect, Detect, Respond, Recover — and prioritise gaps in detection and response.",
        "2. Implement defence-in-depth by layering technical controls (MFA, EDR, WAF) with administrative controls (policy, training) and physical controls, ensuring no single point of failure.",
        "3. Establish a continuous improvement cycle: regularly review SIEM alerts, update threat intelligence feeds, and run tabletop exercises to validate incident response playbooks.",
    ],
}

def _guess_topic(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["phish", "spear", "bec", "email fraud"]):
        return "phishing"
    if any(w in q for w in ["incident", "response", "ransomware", "breach", "contain"]):
        return "incident_response"
    if any(w in q for w in ["vulnerab", "patch", "cve", "scan", "exploit"]):
        return "vulnerability_management"
    if any(w in q for w in ["cloud", "aws", "azure", "gcp", "s3"]):
        return "cloud_security"
    return "default"


class MockGenerator:
    """Returns pre-crafted guidelines for demo/testing without an LLM."""

    def generate(self, query: str, context: str) -> Dict:
        topic = _guess_topic(query)
        guidelines = "\n".join(MOCK_GUIDELINES.get(topic, MOCK_GUIDELINES["default"]))
        return {
            "output":    guidelines,
            "model":     "mock-generator",
            "latency_s": 0.01,
            "tokens":    -1,
        }


# ---------------------------------------------------------------------------
# Auto-selecting generator factory
# ---------------------------------------------------------------------------

def get_generator(prefer: str = "ollama") -> object:
    """
    Return the best available generator in priority order:
      ollama (Mistral 7B local) → mock (demo fallback).

    The HuggingFace pipeline is left as a manual option because downloading
    Mistral 7B (~13 GB) is impractical in automated environments.
    """
    if prefer == "ollama":
        gen = OllamaGenerator()
        if gen.is_available():
            print("[Generator] Using Ollama (Mistral 7B).")
            return gen
        print("[Generator] Ollama not available. Falling back to Mock generator.")
        print("            → To use Mistral: install Ollama and run 'ollama pull mistral'")
    print("[Generator] Using MockGenerator for demonstration.")
    return MockGenerator()


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def parse_guidelines(raw_output: str) -> List[str]:
    """
    Extract numbered guidelines from raw LLM output.
    Handles common formatting variations.
    """
    lines = raw_output.strip().split("\n")
    guidelines = []
    current = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.match(r"^[1-3][\.\)]\s", line):
            if current:
                guidelines.append(current.strip())
            current = line
        else:
            current = current + " " + line if current else line
    if current:
        guidelines.append(current.strip())
    # Ensure exactly 3 items (pad if LLM returned fewer)
    while len(guidelines) < 3:
        guidelines.append(f"{len(guidelines)+1}. [No additional guideline generated]")
    return guidelines[:3]
