"""Local-only populated UI fixture for manual screenshot QA. Never calls model services."""

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"

PROCESS_STATE = {
    "process_name": "Northstar quote preparation and discount approval",
    "actors": ["Sales representative", "Sales manager", "Finance approver"],
    "systems": ["CRM", "Pricing catalogue", "Email"],
    "steps": [
        "Sales confirms the customer and billing entity in CRM.",
        "Sales builds a draft quote from catalogue pricing.",
        "Discounted quotes are emailed to the sales manager for approval.",
        "Sales re-enters the approved terms and sends the quote to the customer.",
    ],
    "decisions": ["Whether a discount requires approval", "Whether customer data is complete"],
    "pain_points": ["Approval waits in an email inbox", "Quote details are entered twice"],
    "evidence": [
        {"source_type": "user", "source": "discovery interview", "claim": "The sales manager reviews discounts above 15%."},
        {"source_type": "document", "source": "discount_policy.md", "document_title": "Commercial Discount Policy", "chunk_id": "discount-policy-a71f", "claim": "Manager approval is required for discounts above 10%."},
        {"source_type": "mcp", "source": "crm_get_customer", "claim": "The CRM exposes customer ownership and billing details."},
    ],
    "conflicts": [{"topic": "Discount approval threshold", "first_claim": "Approval starts above 15%.", "first_source": "User interview", "second_claim": "Approval is required above 10%.", "second_source": "Commercial Discount Policy", "status": "unresolved"}],
    "unknowns": ["Does the CRM provide a supported quote write API?"],
}

RESULT = {
    "process_state_fingerprint": "preview-fingerprint", "model": "gpt-6-luna", "created_at": "2026-10-03T09:00:00Z", "stale": False,
    "analysis": {
        "summary": "Quote preparation is slowed by an email approval handoff and duplicate entry after approval.",
        "findings": [{"finding_id": "finding-1", "category": "manual_handoff", "description": "Discount approval is handed from Sales to a manager by email.", "basis": "reported_fact", "supporting_evidence_ids": ["evidence-001"], "related_steps": ["Discounted quotes are emailed for approval."]}],
        "bottlenecks": ["Approval waits in the manager's email inbox"],
        "manual_handoffs": ["Sales sends discounted quotes to the manager by email"],
        "duplicate_work": ["Approved commercial terms are re-entered"],
        "policy_practice_gaps": ["Reported 15% threshold differs from the documented 10% threshold"],
        "automation_candidates": ["Route approval requests with structured quote data"],
        "unresolved_questions": ["Confirm whether a supported quote write API exists"],
    },
    "proposal": {
        "objective": "Reduce approval delay and duplicate entry while preserving explicit human approval.",
        "recommendations": [
            {"recommendation_id": "recommendation-1", "title": "Route structured discount approvals", "description": "Create a structured approval request when the documented threshold is crossed.", "addresses_finding_ids": ["finding-1"], "proposed_automation": "Package quote context and route it to the correct manager.", "human_control": "The manager remains responsible for approval.", "dependencies": ["Resolve the policy/practice threshold conflict"], "risks": ["Incorrect routing if account ownership is stale"], "supporting_evidence_ids": ["evidence-001", "evidence-002"]},
            {"recommendation_id": "recommendation-2", "title": "Write approved terms back to the quote", "description": "Remove duplicate entry after approval.", "addresses_finding_ids": ["finding-1"], "proposed_automation": "Update the quote record after approval.", "human_control": "Sales confirms the final customer-facing quote.", "dependencies": ["Supported CRM write API"], "risks": ["Unverified write capability"], "supporting_evidence_ids": ["evidence-003"]},
        ],
        "to_be_steps": [
            {"order": 1, "description": "Sales confirms customer and quote inputs.", "automation_level": "human_controlled", "human_owner": "Sales representative"},
            {"order": 2, "description": "The system applies the documented approval rule.", "automation_level": "automated", "human_owner": None},
            {"order": 3, "description": "The manager reviews the structured request.", "automation_level": "hybrid", "human_owner": "Sales manager"},
            {"order": 4, "description": "Sales confirms and sends the final quote.", "automation_level": "human_controlled", "human_owner": "Sales representative"},
        ],
        "retained_human_decisions": ["Approve exceptional discounts", "Confirm the final customer-facing quote"],
        "assumptions": ["The documented threshold is adopted after the conflict is resolved"],
    },
    "verification": {
        "verifications": [
            {"recommendation_id": "recommendation-1", "status": "partially_supported", "explanation": "The routing need is supported, but the applicable threshold is unresolved.", "supporting_evidence_ids": ["evidence-001", "evidence-002"], "missing_information": ["Resolve the approval threshold"]},
            {"recommendation_id": "recommendation-2", "status": "blocked_by_unknown", "explanation": "A supported CRM write capability has not been evidenced.", "supporting_evidence_ids": ["evidence-003"], "missing_information": ["Confirm a supported quote write API"]},
        ],
        "approved_recommendation_ids": [], "rejected_recommendation_ids": [], "unresolved_recommendation_ids": ["recommendation-1", "recommendation-2"],
    },
}

AUTO_RUN = """
<script>
document.documentElement.dataset.previewWidth = String(window.innerWidth);
document.documentElement.dataset.previewScrollWidth = String(document.documentElement.scrollWidth);
window.addEventListener('load', () => {
  const previewParams = new URLSearchParams(window.location.search);
  const guideOnly = previewParams.has('guide');
  if (guideOnly) {
    document.querySelector('#demo-guide').open = true;
    setTimeout(() => { document.body.dataset.horizontalOverflow = String(document.documentElement.scrollWidth > window.innerWidth); }, 300);
    return;
  }
  if (previewParams.has('catalog')) {
    document.querySelector('#rag-status').open = true;
    return;
  }
  const send = (value) => {
    const input = document.querySelector('#message-input');
    input.value = value;
    input.dispatchEvent(new Event('input'));
    document.querySelector('#chat-form').requestSubmit();
  };
  setTimeout(() => send('My sales team takes too long to build quotes.'), 120);
  setTimeout(() => send('We email discounts above 15% to the manager, then enter the approved terms again.'), 450);
  setTimeout(() => document.querySelector('#analyze-process').click(), 900);
  setTimeout(() => document.querySelector('.recommendation-disclosure summary')?.click(), 1300);
  setTimeout(() => { document.body.dataset.horizontalOverflow = String(document.documentElement.scrollWidth > window.innerWidth); }, 1500);
});
</script>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def send_bytes(self, data: bytes, content_type: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, value):
        self.send_bytes(json.dumps(value).encode(), "application/json")

    def send_sse(self, events):
        payload = "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()
        self.send_bytes(payload, "text/event-stream")

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/mobile-preview":
            wrapper = '<!doctype html><meta charset="utf-8"><title>390px preview</title><style>html,body{margin:0;background:#dfe4eb}iframe{display:block;width:390px;height:5200px;border:0;background:white}</style><iframe src="/index.html" title="390px application preview"></iframe>'
            return self.send_bytes(wrapper.encode(), "text/html; charset=utf-8")
        if parsed.path in {"/", "/index.html"}:
            html = (FRONTEND / "index.html").read_text(encoding="utf-8").replace("</body>", AUTO_RUN + "</body>")
            return self.send_bytes(html.encode(), "text/html; charset=utf-8")
        if parsed.path in {"/styles.css", "/app.js"}:
            path = FRONTEND / parsed.path.lstrip("/")
            return self.send_bytes(path.read_bytes(), "text/css" if path.suffix == ".css" else "text/javascript")
        if parsed.path == "/runtime-status": return self.send_json({"model": "gpt-6-luna"})
        if parsed.path == "/mcp-status": return self.send_json({"connected": True, "tools_discovered": 2, "tools": [{"name": "crm_get_customer", "description": "Look up a customer in the company's CRM by customer name."}, {"name": "pricing_get_product", "description": "Look up product details and standard price by SKU."}], "server": "Northstar Business Systems", "transport": "stdio"})
        if parsed.path == "/rag-status": return self.send_json({"available": True, "embedding_model": "text-embedding-3-small", "documents": 3, "chunks": 14, "stale": False})
        if parsed.path == "/knowledge-catalog": return self.send_json({"available": True, "documents": [{"title": "Commercial Discount Policy", "filename": "discount_policy.md", "chunk_count": 4, "topics": "discount approval rules, approver roles, required controls"}, {"title": "Northstar Quote Process", "filename": "quote_process.md", "chunk_count": 5, "topics": "quote preparation, approvals, issuance and process recording"}, {"title": "Product Operations Guidelines", "filename": "product_guidelines.md", "chunk_count": 5, "topics": "operational product guidance and quote-related handling"}]})
        if parsed.path == "/process-state": return self.send_json(PROCESS_STATE)
        if parsed.path == "/analysis": return self.send_json({"status": "not_available", "result": None, "stale": False})
        if parsed.path == "/evaluation-report":
            report = {"created_at": "2026-10-03T08:30:00Z", "model": "gpt-6-luna", "eval_model": "gpt-6-luna", "selected_scenario_count": 10, "scenario_execution_count": 10, "runs_per_scenario": 1, "scenario_runs_passed": 9, "scenario_runs_total": 10, "metrics": [{"metric": "Conflict preservation", "passed": 10, "total": 10, "pass_rate": 1}, {"metric": "Verifier calibration", "passed": 9, "total": 10, "pass_rate": .9}], "results": [{"scenario_id": "quote-discovery", "scenario_name": "Quote process discovery", "run_number": 1, "passed": True, "criteria": [{"evaluator": "deterministic", "skipped": False, "passed": True, "description": "Captures steps", "explanation": "Captured"}]}, {"scenario_id": "unsupported-write", "scenario_name": "Verifier blocks unsupported write", "run_number": 1, "passed": False, "criteria": [{"evaluator": "semantic", "skipped": False, "passed": False, "description": "Blocks unsupported capability", "explanation": "The dependency needs clearer evidence."}]}]}
            return self.send_json({"status": "available", "report": report})
        if parsed.path == "/chat-stream":
            message = parse_qs(parsed.query).get("message", [""])[0]
            events = [{"type": "llm_call_started", "technology": "OpenAI Responses API", "model": "gpt-6-luna", "operation": "discovery", "purpose": "Interpret process context"}, {"type": "llm_call_completed", "technology": "OpenAI Responses API", "model": "gpt-6-luna", "operation": "discovery", "status": "success"}]
            if "15%" in message:
                events += [{"type": "rag_retrieval_started", "technology": "Application-owned RAG", "query": "discount approval policy", "top_k": 3}, {"type": "rag_retrieval_completed", "technology": "Plain Python cosine similarity", "status": "success", "query": "discount approval policy", "embedding_model": "text-embedding-3-small", "index_chunk_count": 14, "top_k": 3, "results": [{"document_title": "Commercial Discount Policy", "chunk_id": "discount-policy-a71f", "similarity_score": .94}]}, {"type": "rag_evidence_supplied", "technology": "OpenAI Responses API", "model": "gpt-6-luna", "documents": ["Commercial Discount Policy"], "chunk_ids": ["discount-policy-a71f"]}, {"type": "assistant_message", "content": "I found a **policy/practice conflict**:\n\n- Reported practice: approval above **15%**\n- Documented policy: approval above **10%**\n\nI preserved both as unresolved evidence."}, {"type": "process_state_updated", "patch": {"conflicts_to_add": [{}], "evidence_to_add": [{}, {}]}, "state": PROCESS_STATE}, {"type": "interaction_completed", "status": "success"}]
            else:
                events += [{"type": "assistant_message", "content": "Let's map the current quote process.\n\n1. Who builds the quote?\n2. Which systems hold customer and pricing data?\n3. Where does approval slow down?"}, {"type": "interaction_completed", "status": "success"}]
            return self.send_sse(events)
        return self.send_bytes(b"Not found", "text/plain", 404)

    def do_POST(self):
        if self.path == "/analyze-process":
            events = [{"type": "orchestration_started", "model": "gpt-6-luna", "stage": "orchestration", "snapshot_counts": {"steps": 4, "evidence": 3}}, {"type": "analysis_started", "model": "gpt-6-luna", "stage": "analysis"}, {"type": "analysis_completed", "model": "gpt-6-luna", "stage": "analysis", "status": "success", "finding_count": 4}, {"type": "automation_design_started", "model": "gpt-6-luna", "stage": "automation_design"}, {"type": "automation_design_completed", "model": "gpt-6-luna", "stage": "automation_design", "status": "success", "recommendation_count": 2, "retained_human_decision_count": 2}, {"type": "verification_started", "model": "gpt-6-luna", "stage": "verification"}, {"type": "verification_completed", "model": "gpt-6-luna", "stage": "verification", "status": "success", "status_counts": {"supported": 0, "partially_supported": 1, "unsupported": 0, "blocked_by_unknown": 1}}, {"type": "orchestration_completed", "model": "gpt-6-luna", "stage": "orchestration", "status": "success", "result": RESULT}]
            return self.send_sse(events)
        if self.path == "/reset-discovery": return self.send_json({"state": {}})
        return self.send_bytes(b"Not found", "text/plain", 404)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8766), Handler)
    print("Visual preview: http://127.0.0.1:8766", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
