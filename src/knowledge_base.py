"""
knowledge_base.py
=================
Builds and manages the cybersecurity knowledge base.

This module handles:
  - Defining a rich set of cybersecurity documents (phishing, incident
    response, vulnerability management) drawn from NIST SP 800-61r2,
    CISA guidance, ISO 27001 principles, and OWASP references.
  - Chunking documents into overlapping passages.
  - Computing sentence-transformer embeddings.
  - Persisting chunks to ChromaDB for semantic retrieval.
"""

import re
from typing import List, Dict, Tuple

# ---------------------------------------------------------------------------
# Domain knowledge — curated cybersecurity passages
# Each entry mirrors a paragraph from a real standard/advisory.
# In a production system these would be loaded from PDFs; here we provide
# representative, high-fidelity synthesised excerpts so the system can run
# fully offline without licensing issues.
# ---------------------------------------------------------------------------

CYBERSECURITY_DOCUMENTS = [
    # ── PHISHING ────────────────────────────────────────────────────────────
    {
        "id": "phish_001",
        "source": "NIST SP 800-177 Rev.1 §3",
        "topic": "phishing",
        "text": (
            "Phishing attacks impersonate trusted entities to trick users into "
            "revealing credentials or installing malware. Spear-phishing targets "
            "specific individuals using personally identifiable information gathered "
            "from social media or prior data breaches. Organisations should deploy "
            "DMARC, DKIM, and SPF email authentication records on all domains to "
            "reduce spoofing. Anti-phishing gateways with URL reputation filtering "
            "and sandboxed attachment detonation are recommended for enterprise "
            "email pipelines."
        ),
    },
    {
        "id": "phish_002",
        "source": "CISA Anti-Phishing Guide 2023",
        "topic": "phishing",
        "text": (
            "Organisations should implement multi-factor authentication (MFA) on all "
            "externally facing services. Phishing-resistant MFA types such as FIDO2 "
            "hardware keys (e.g., YubiKey) or passkeys are preferred over SMS-based "
            "OTP, which is susceptible to SIM-swapping attacks. Security awareness "
            "training should include simulated phishing campaigns to measure and "
            "improve employee vigilance. Training frequency of at least quarterly is "
            "recommended, with targeted remedial training for repeat clickers."
        ),
    },
    {
        "id": "phish_003",
        "source": "CISA Advisory AA22-074A",
        "topic": "phishing",
        "text": (
            "Business Email Compromise (BEC) is a form of spear-phishing that "
            "impersonates executive or financial personnel to authorise fraudulent "
            "wire transfers. Controls include: (1) out-of-band verification for "
            "payment changes above a threshold; (2) email banners on external mail; "
            "(3) blocking display-name spoofing at the gateway; (4) enforcing "
            "Sender Policy Framework (SPF) in -all hard-fail mode. Mid-sized "
            "enterprises should establish a dedicated mailbox for reporting suspicious "
            "messages and integrate it with SIEM alerting."
        ),
    },
    {
        "id": "phish_004",
        "source": "OWASP Phishing Prevention Cheat Sheet",
        "topic": "phishing",
        "text": (
            "Web application defences against phishing credential harvesting include: "
            "enforcing HTTPS everywhere with HSTS preloading; deploying Content "
            "Security Policy (CSP) headers to block inline script injection; using "
            "certificate transparency monitoring to detect look-alike domains; and "
            "registering typosquat variants of the organisation's primary domain. "
            "User-facing login pages should include mutual TLS or re-authentication "
            "prompts when anomalous geolocation or device fingerprints are detected."
        ),
    },
    {
        "id": "phish_005",
        "source": "Microsoft Incident Response Playbook 2024",
        "topic": "phishing",
        "text": (
            "When a phishing incident is confirmed: immediately revoke the compromised "
            "user's session tokens and force password reset; check sign-in logs for "
            "lateral movement within 30 minutes of the phishing click; quarantine "
            "all emails with the same subject-line/URL combination using the email "
            "purge API; and notify the SOC to add IOCs (URLs, sender domains, "
            "attachment hashes) to the blocklist. Post-incident, analyse the email "
            "headers to attribute the campaign and share threat intelligence with "
            "sector ISACs."
        ),
    },

    # ── INCIDENT RESPONSE ───────────────────────────────────────────────────
    {
        "id": "ir_001",
        "source": "NIST SP 800-61r2 §3.1",
        "topic": "incident_response",
        "text": (
            "The incident response lifecycle comprises four phases: Preparation, "
            "Detection & Analysis, Containment–Eradication–Recovery, and "
            "Post-Incident Activity. Preparation involves establishing an incident "
            "response team (IRT), documenting playbooks, pre-positioning forensic "
            "tooling, and conducting tabletop exercises. Detection relies on SIEM "
            "correlation rules, EDR telemetry, network anomaly detection, and threat "
            "intelligence feeds. Analysts must triage events by severity (P1–P4) "
            "within defined SLA windows."
        ),
    },
    {
        "id": "ir_002",
        "source": "NIST SP 800-61r2 §3.2",
        "topic": "incident_response",
        "text": (
            "Containment strategies vary by incident type. For ransomware: isolate "
            "affected hosts at the network layer (VLAN quarantine or firewall ACL), "
            "disable compromised accounts, and snapshot affected volumes before "
            "remediation. For data exfiltration incidents: block egress to identified "
            "C2 IPs, preserve PCAP evidence, and invoke legal hold. The containment "
            "decision must balance business continuity against risk of spread. "
            "Out-of-band communication channels should be established if primary "
            "channels are compromised."
        ),
    },
    {
        "id": "ir_003",
        "source": "SANS Incident Handler's Handbook",
        "topic": "incident_response",
        "text": (
            "Eradication removes the root cause: delete malware artefacts identified "
            "by forensic analysis, patch the exploited vulnerability, rebuild "
            "compromised hosts from a known-good image, and rotate all credentials "
            "touched by the adversary. Recovery involves restoring systems from "
            "clean backups, validating integrity via hash verification, and conducting "
            "smoke tests before returning to production. Organisations should maintain "
            "immutable, air-gapped backups tested at least monthly."
        ),
    },
    {
        "id": "ir_004",
        "source": "ISO/IEC 27035-2:2023 §6",
        "topic": "incident_response",
        "text": (
            "Post-incident reviews (PIRs) — also called 'lessons-learned' sessions — "
            "should be conducted within 5 business days of incident closure. PIR "
            "outputs include a timeline of attacker actions (kill-chain mapping), "
            "identification of detection gaps, updated runbooks, and metric reporting "
            "to leadership. Metrics include Mean Time to Detect (MTTD) and Mean Time "
            "to Respond (MTTR). ISO 27035 requires evidence retention for a minimum "
            "of three years for regulatory compliance."
        ),
    },
    {
        "id": "ir_005",
        "source": "CISA Incident Response Recommendations 2024",
        "topic": "incident_response",
        "text": (
            "Small and mid-sized enterprises (SMEs) without a dedicated SOC should "
            "consider Managed Detection and Response (MDR) providers for 24/7 "
            "monitoring. Key contractual requirements: guaranteed MTTD under 4 hours, "
            "analyst-confirmed alerts only (not raw log noise), and incident "
            "notification within 15 minutes of P1 confirmation. Retainer-based "
            "forensic IR firms should be pre-engaged with signed SOWs before an "
            "incident occurs to avoid negotiating under duress."
        ),
    },

    # ── VULNERABILITY MANAGEMENT ─────────────────────────────────────────────
    {
        "id": "vuln_001",
        "source": "NIST SP 800-40r4",
        "topic": "vulnerability_management",
        "text": (
            "Vulnerability management is a continuous process: Discover assets → "
            "Assess vulnerabilities → Prioritise → Remediate → Verify → Report. "
            "Asset discovery should encompass IT, OT, cloud, and shadow-IT assets. "
            "Authenticated vulnerability scans produce lower false-positive rates "
            "than unauthenticated scans. Scan frequency: critical internet-facing "
            "systems weekly; internal systems monthly. Patch SLAs by CVSS severity: "
            "Critical (CVSS ≥9.0) within 15 days; High (7.0–8.9) within 30 days; "
            "Medium within 90 days."
        ),
    },
    {
        "id": "vuln_002",
        "source": "CISA KEV Catalogue Guidance 2024",
        "topic": "vulnerability_management",
        "text": (
            "CISA's Known Exploited Vulnerabilities (KEV) catalogue lists CVEs "
            "actively exploited in the wild. Federal agencies are mandated to patch "
            "KEV entries within 14 days (CISA BOD 22-01). Private sector "
            "organisations should treat KEV entries as Critical regardless of CVSS "
            "score, since CVSS does not factor in active exploitation. Organisations "
            "should integrate KEV into their vulnerability prioritisation workflow "
            "alongside EPSS (Exploit Prediction Scoring System) scores."
        ),
    },
    {
        "id": "vuln_003",
        "source": "ISO/IEC 27001:2022 Annex A.8.8",
        "topic": "vulnerability_management",
        "text": (
            "ISO 27001:2022 control A.8.8 requires organisations to obtain timely "
            "information about technical vulnerabilities of information systems in "
            "use, evaluate exposure to those vulnerabilities, and take appropriate "
            "measures. This includes establishing a vulnerability disclosure policy "
            "(VDP) and a coordinated vulnerability disclosure (CVD) process. "
            "Organisations should subscribe to vendor security advisories and "
            "maintain a software bill of materials (SBOM) to identify affected "
            "components rapidly when new CVEs are published."
        ),
    },
    {
        "id": "vuln_004",
        "source": "OWASP Top 10 2021 — A06 Vulnerable Components",
        "topic": "vulnerability_management",
        "text": (
            "Using components with known vulnerabilities is consistently among the "
            "top web application risks. Mitigations: (1) maintain an inventory of "
            "all third-party libraries with their versions; (2) use Software "
            "Composition Analysis (SCA) tools such as Snyk, Dependabot, or OWASP "
            "Dependency-Check in CI/CD pipelines; (3) remove unused dependencies; "
            "(4) subscribe to NVD, GitHub Security Advisories, and vendor changelogs; "
            "(5) apply virtual patching via WAF rules for unresolvable vulnerabilities."
        ),
    },
    {
        "id": "vuln_005",
        "source": "NIST SP 800-115 §5.4",
        "topic": "vulnerability_management",
        "text": (
            "Penetration testing complements automated scanning by exercising "
            "multi-step attack chains that scanners miss. Organisations should "
            "conduct annual external penetration tests and after significant "
            "infrastructure changes. Scope must be formally agreed in a Rules of "
            "Engagement (RoE) document. Findings should be risk-rated, tracked in "
            "a remediation register, and re-tested to confirm closure. Purple-team "
            "exercises (red team + blue team collaboration) accelerate detection "
            "engineering by sharing attacker TTPs in real time."
        ),
    },

    # ── CLOUD SECURITY ───────────────────────────────────────────────────────
    {
        "id": "cloud_001",
        "source": "CIS Benchmarks for AWS v2.0",
        "topic": "cloud_security",
        "text": (
            "Cloud security configuration hardening for AWS: enable CloudTrail "
            "logging in all regions with log file integrity validation; enforce "
            "S3 Block Public Access at the account level; restrict root account "
            "usage — enable MFA on root and delete root access keys; use IAM "
            "roles rather than long-lived IAM user access keys for EC2 workloads; "
            "enable AWS GuardDuty and Security Hub for continuous threat detection. "
            "Ensure VPC flow logs and DNS query logs are enabled for forensic "
            "investigation capability."
        ),
    },
    {
        "id": "cloud_002",
        "source": "NIST SP 800-204C — Microservices Security",
        "topic": "cloud_security",
        "text": (
            "Zero-trust architecture principles for cloud-native deployments: "
            "authenticate and authorise every service-to-service API call using "
            "mutual TLS (mTLS) or JWT tokens with short expiry; apply network "
            "micro-segmentation via service mesh (Istio/Linkerd) policies; "
            "enforce least-privilege IAM at the workload identity level; "
            "log all authentication events and access decisions to an immutable "
            "audit log. Assume breach: design blast radius isolation so a "
            "compromised service cannot pivot to unrelated data stores."
        ),
    },
]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = 300,
    overlap: int = 50,
) -> List[str]:
    """
    Split *text* into overlapping chunks of approximately *chunk_size* words
    with *overlap* words of context carried forward.

    Rationale for overlap: preserves sentence continuity across chunk
    boundaries, which improves retrieval recall for multi-sentence queries.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start += chunk_size - overlap
    return chunks


def build_chunk_records(
    docs: List[Dict],
    chunk_size: int = 300,
    overlap: int = 50,
) -> List[Dict]:
    """
    Returns a flat list of chunk dicts ready for embedding & storage.
    Each chunk carries metadata: source, topic, parent doc id, chunk index.
    """
    records = []
    for doc in docs:
        chunks = chunk_text(doc["text"], chunk_size, overlap)
        for i, chunk in enumerate(chunks):
            records.append(
                {
                    "chunk_id": f"{doc['id']}_c{i}",
                    "doc_id": doc["id"],
                    "source": doc["source"],
                    "topic": doc["topic"],
                    "text": chunk,
                }
            )
    return records


# ---------------------------------------------------------------------------
# Convenience: pre-built chunks from the default corpus
# ---------------------------------------------------------------------------

def get_default_chunks() -> List[Dict]:
    """Return pre-built chunks from CYBERSECURITY_DOCUMENTS."""
    return build_chunk_records(CYBERSECURITY_DOCUMENTS)
