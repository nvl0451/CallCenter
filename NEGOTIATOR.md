# Sales Negotiator — Operating Guide (Draft)

This note describes how the sales assistant should drive conversations from first contact to purchase with a clear split of responsibilities between fast classification and a dialog model orchestrating tools.

## 1) Overview

- Two gates up front using a lightweight classifier:
  - Gate A: intent → sales vs. everything else
  - Gate B: plan specificity → Basic | Plus | Business | unsure
- Two paths after the gates:
  - Decisive path (user already knows the plan): move quickly to a grounded pitch and checkout.
  - Guided path (user is unsure): ask 1–2 questions, recommend a plan, then move to checkout.
- The dialog model keeps the voice and chooses when to consult tools. The server validates tool calls and enforces budgets.

## 2) Session State (slots)

Track compact facts to guide the next step:

- intent, plan_candidate, plan_confidence
- seats, channels[], sso_required?, analytics_required?
- budget_sensitivity, discount_request
- decision_status (considering | ready_to_buy | need_approval)
- admin_email
- objection (price | feature | security | timing)
- out_of_scope

The model reads/writes these via short summaries; the server can expose them for observability.

## 3) Tools (thin, JSON I/O)

The dialog model may request tools; the server executes them and returns strict JSON.

- rag_search(query, top_k): retrieve relevant plan text/snippets (include {source,title}).
- plan_compare(a,b): crisp deltas (features, limits, price lines).
- price_calc(plan, seats, billing=monthly|annual, discount?%): totals and line‑items.
- discount_policy(context): allowed concessions and conditions (if any).
- create_quote(plan, seats, billing, admin_email, discount?): returns quote_id/link.
- schedule_call(pref): calendar slots for follow‑up.
- compliance_info(plan): security/SLA bullets for objections.

Server budgets: ≤3 tool calls / turn, ≤8s per tool, ≤20s total. Reject invalid or out‑of‑policy calls and ask the model to elicit missing inputs.

## 4) Path Selection

- Decisive path when:
  - intent = sales AND plan_confidence ≥ threshold (e.g., 0.70)
  - no hesitation markers in the user text
- Guided path when:
  - plan_confidence below threshold OR hesitation markers present
  - user asks comparisons, security questions, or starts negotiating

Handover from guided → decisive after the model has seats + channels (+ SSO if needed) and proposes a plan with a short rationale.

## 5) Dialogue Guidelines (RU)

- Tone: тёплый, дружелюбный, уверенный. Короткие, человеческие фразы.
- RU‑only; без эмодзи и иностранных символов.
- Ask ≤2 уточняющих вопроса за ход.
- Decisive path: brief greeting → price (1 line) → 3–5 выгод плана (без ссылок на другие планы) → мягкий призыв к действию (выбор периода оплаты, e‑mail администратора для приглашения команды).
- Guided path: brief greeting → 2 коротких вопроса (seats, каналы; опционально SSO/аналитика) → рекомендация плана → мягкий призыв к действию.

## 6) Handling Questions, Haggling, and Out‑of‑Scope

- Feature/plan questions: use plan_compare or rag_search, answer кратко, затем один уточняющий вопрос и предложение продолжить.
- Price pushback: detect discount intent; consult discount_policy. Если допустимо — озвучить прозрачную скидку с условиями (объём/год). Если нет — сфокусировать на ценности, предложить годовую оплату/пробный период/созвон.
- Security/SLA concerns: use compliance_info and keep it factual; затем вернуть к выбору плана.
- Out‑of‑scope (e.g., unrelated purchases): вежливо переориентировать на тарифы и задать один фокусный вопрос.

## 7) Guardrails

- Никогда не выдумывать скидки/функции: только через discount_policy и RAG.
- Не собирать платёжные данные в чате. Использовать create_quote и e‑mail администратора для онбординга команды.
- Минимизировать PII: только e‑mail администратора и необходимое для расчёта.
- Лимиты по инструментам и времени соблюдать; при исчерпании — краткий ответ и предложение продолжить в следующем сообщении.

## 8) Observability

Log per turn:

- stage: decisive | guided | handoff
- tools_used: ordered list with {tool, ms, ok}
- slots_changed: which fields were updated
- latencies: classify_ms, rag_ms, llm_ms, total_ms
- outcomes: quote_created?, quote_accepted?

## 9) Test Scenarios (smoke)

- Decisive purchase (e.g., Plus) → confirm billing + e‑mail → quote created.
- Guided discovery → recommend plan → confirm → quote created.
- Plan comparison loop (Plus vs Business) → concise deltas → proceed.
- Price negotiation: allowed vs. not allowed → outcomes.
- Security objection: compliance_info → proceed.
- Out‑of‑scope message → polite redirect → proceed.
