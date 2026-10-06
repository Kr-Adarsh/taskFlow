# Validation results

## Live acceptance

All three end-to-end demos passed using real Groq `openai/gpt-oss-120b` planning and decisions, with model-generated Python for sales analysis. The same runtime and tools handled each objective:

| Case | Result | Decisions | API calls | Run duration |
| --- | --- | ---: | ---: | ---: |
| Invoice intake | PASS | 10 | 12 | 418.5s |
| Complaint → CRM → Support | PASS | 9 | 12 | 372.2s |
| Sales analysis | PASS | 4 | 6 | 100.4s |

| Case | Input tokens | Output tokens | Total tokens |
| --- | ---: | ---: | ---: |
| Invoice intake | 48,259 | 6,404 | 54,663 |
| Complaint → CRM → Support | 42,405 | 7,181 | 49,586 |
| Sales analysis | 14,337 | 3,724 | 18,061 |
| **Total** | **105,001** | **17,309** | **122,310** |

Verified outcomes:

- Invoice: exactly one Acme Corp INV-1044, INR 84,500, due 2026-10-15. The injected pre-commit 503 was safely retried.
- Complaint: exactly one High-priority ticket after confirming Enterprise status, with the correct source linkage and a grounded summary.
- Sales: all 800 rows processed locally; South had the largest revenue decline at 27,000, confirmed by independent recomputation.
- All three passed independent verification and the final mutation audit, with no unwanted business changes.

## Software regression tests

| Passed | Failed | Duration |
| ---: | ---: | ---: |
| 412 | 0 | 153.46s |

Covered browser actions, database writes, duplicate protection, failure recovery, context handling, Python isolation, result ownership and independent verification. The real-model end-to-end results are reported separately above.
