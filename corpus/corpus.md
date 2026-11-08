# TaskFlow Corpus Documentation

- **Total entries:** 40 structured FAQ entries in `faqs.json` (`id`, `title`, `body`, `category`, `last_updated`).
- **Categories:** account, billing, integrations, tasks, security, bugs.
- **Deliberate topic overlap with different wording:** `faq-001` / `faq-002` / `faq-003` (password), `faq-006` / `faq-039` (cancel vs. downgrade), `faq-007` / `faq-008` (Slack setup vs. notifications).

## Intentionally unanswerable questions

The first three are the required ones (clearly out of scope). The last two are **in-domain** (they sound like real TaskFlow questions but nothing in the corpus answers them) and are the harder abstention test.

1. "Can I connect my TaskFlow account to a Nintendo Switch or PlayStation console?"
2. "How do I bake a sourdough loaf of bread using the mobile app timer?"
3. "What is the airspeed velocity of an unladen swallow in TaskFlow's enterprise settings?"
4. "Can I change my billing currency from USD to EUR?" (in-domain)
5. "Does TaskFlow have a Gantt chart view?" (in-domain)

> Note: SAML/Okta SSO *is* answered (`faq-014`), so it is deliberately not used as an unanswerable example.
