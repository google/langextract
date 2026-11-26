---
name: confirmed-orders
description: Prepare a plain UTF-8 order ledger for grounded extraction.
---

Read the input as plain text. Keep only lines containing the exact, case-sensitive status label `CONFIRMED:`. Preserve those lines verbatim and in order; omit everything else. Do not infer or summarize orders.
