"""Example 1 — FAQ: store facts, ask questions.

    pip install abm-runtime
    python 01_faq.py
"""

from abm import Memory

mem = Memory()                              # default: 2048-bit trace

mem.store("password_reset", "handled_by", "account_settings_page")
mem.store("refund_request", "handled_by", "support_ticket")
mem.store("invoice_download", "handled_by", "billing_page")
mem.store("support_ticket", "response_time", "24_hours")

answer, confidence = mem.query("refund_request", "handled_by")
print(f"Refunds are handled by: {answer}  (confidence {confidence:.0%})")

# two-step question: where do refunds go, and how fast is that channel?
answer, confidence = mem.chain("refund_request",
                               ["handled_by", "response_time"])
print(f"Refund response time:   {answer}  (confidence {confidence:.0%})")

# is a specific fact in memory? one boolean, no search
print("Invoices on billing page?",
      mem.member("invoice_download", "handled_by", "billing_page"))
