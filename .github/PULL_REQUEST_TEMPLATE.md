## Summary
Brief description of what this PR changes and why.

## Security & Hygiene Checklist
- [ ] Core CLI (`nearai_agy.py`, `attestation_evidence.py`) remains Python 3.9+ stdlib only.
- [ ] No live credentials, private keys, raw production quotes/JWTs, or personal local paths were added.
- [ ] Fail-closed security invariants (URL/redirect checks, proxy bounds, attestation semantics) are preserved.
- [ ] Offline unit tests added/updated and passing (`pytest -v`).
- [ ] Bash forwarders and example scripts pass `bash -n`.
