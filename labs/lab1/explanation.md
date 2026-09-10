# Lab 1 Explanation

This lab is designed to teach a very specific lesson: a model output can fail in two completely different ways.

1. It can fail to parse at all.
2. It can parse successfully but still be semantically invalid.

The point of the lab is to show that both problems matter, and that schema validation is what separates a working extractor from a broken one.

---

## 1. What v0_naive is doing

The file `v0_naive.py` contains the intentionally bad implementation.

Its core function is:

```python
def extract_v0(ticket: str) -> dict:
    out = chat(PROMPT.format(ticket=ticket), tier="SMALL", max_tokens=300)
    return json.loads(out)
```

This is a naive pattern: it assumes the model will always return valid raw JSON.

In reality, the model is configured to return fenced output, usually like this:

```json
```json
{"category": "billing", "urgency": "high", ...}
```
```

`json.loads()` cannot parse that, because the string contains Markdown fences and extra wrapper text. That is why the lab reports `markdown_fence` under Part 1.

---

## 2. Why the parse fails

The script explicitly detects the reason for failure using `classify_parse_failure()`.

The key check is:

```python
if FENCE.search(raw):
    return "markdown_fence"
```

This means the model output is not raw JSON. It is fenced as Markdown, which is common when the model is told to return code blocks.

That single parse failure blocks the pipeline. Nothing else can be processed because the target object is never extracted.

---

## 3. What is hiding behind the parse failure

The script has a second phase called `salvage()`.

This helper strips the wrapper and then searches for the actual JSON object inside the response. It does something like this:

```python
text = raw.strip()
if FENCE.search(text):
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text.strip())
start, end = text.find("{"), text.rfind("}")
obj = json.loads(text[start : end + 1])
```

This recovers the valid JSON object even when the response was fenced.

This reveals the second layer of problems: the model may produce parseable JSON that is still wrong.

---

## 4. The semantic errors inside valid JSON

After the JSON is recovered, the script checks for content problems using `content_problems()`.

It looks for things like:

- `urgency_is_string`
- `category_out_of_set`
- missing required fields
- policy_number invented
- other schema violations

For example, the model may produce:

```json
{"urgency": "high", "category": "not_a_valid_category"}
```

This is technically valid JSON, but it violates the expected schema.

In the lab output, the main bad values are:

- `urgency` is a string instead of an integer
- `category` is outside the allowed set

This is exactly the point of Part B: parseable is not the same as correct.

---

## 5. Why the one-line fix is not enough

The lab says the “one-line fix” is to stop parsing the raw fenced response and instead parse the JSON body inside it.

That change fixes the parse step:

- before: 0/2 parsed
- after: 2/2 parsed

But the lab also warns that this still does not produce clean records.

The next metric is:

- clean records: 0/2

That means the JSON is parseable, but still invalid according to the business contract.

This is the entire justification for Part B.

---

## 6. Why Part B is necessary

The `extract.py` file is the real implementation that follows the lab’s learning path.

It introduces a schema using Pydantic with fields such as:

- category
- urgency
- sentiment
- product
- language
- policy_number
- evidence

The schema constrains what the model is allowed to return. This is important because it prevents:

- invalid categories
- urgency values as strings
- invented policy numbers
- non-standard output values

The model decides some of this, but code decides the deterministic parts.

This separation is the key idea behind the lab.

---

## 7. What the lab is teaching

The lab teaches this lifecycle:

1. The model emits text.
2. The code tries to parse JSON.
3. It strips wrappers if needed.
4. It validates the schema.
5. It repairs or flags invalid data.
6. It moves deterministic logic into code rather than leaving it to the model.

The endpoint is not just “parse JSON.” The endpoint is:

- parse safely
- validate strictly
- reject or repair invalid outputs
- keep deterministic logic out of the model

---

## 8. The core lesson in one sentence

A model output is only useful if it is both parseable and valid under the schema expected by the application.

---

## 9. Quick reference

Files involved:

- `v0_naive.py` — intentionally broken extractor showing the failure modes
- `extract.py` — Part B/C implementation with schema and validation
- `run_eval.py` — evaluation harness that checks extractor quality

Major concepts:

- `markdown_fence` — output wrapped in code fences
- `urgency_is_string` — wrong type
- `category_out_of_set` — category not allowed
- `policy_number_invented` — fabricated data
- schema validation — ensures output matches contract

---

## 10. Final takeaway

The lab is not just about parsing JSON. It is about understanding that model outputs are unreliable unless they are:

- parsed correctly,
- constrained by schema,
- checked for business validity,
- and hardened for production use.

---

# Answers to the questions at the end of `v0_naive.py`

### 1. Which rows are in the T1 §3 taxonomy, and which two are not?

Most of the rows map cleanly to the taxonomy:

- `json wrapped in a markdown fence`, `extra prose before or after the JSON`, and `not valid JSON at all` are all forms of failure #5, `Malformed output`.
- `valid JSON, missing a required field`, `valid JSON, category outside the allowed set`, and `urgency as a string instead of an int` are all failure #6, `Schema violation`.
- `policy number invented (not present in the text)` is failure #8, `Hallucination`.

The two rows that do not have their own dedicated entry in T1 §3 are:

1. `Unhandled exception` — this is not one of the nine LLM-call failure modes; it belongs to application robustness / error handling.
2. `Not valid JSON at all` — this is not a separate numbered failure in the taxonomy, but a subtype of #5 (`Malformed output`).

So the lab’s intended reading is: the taxonomy covers the model-output failure modes, while `unhandled exception` is a software engineering failure that sits outside that taxonomy.

### 2. Why is the second number (`ok_if_fenceless`) the whole justification for Part B?

Because it shows the core distinction between:

- `parsed successfully`, and
- `clean / valid for downstream use`.

The one-line fix in `extract_v0` improves the parsing rate, but it does not solve the semantic defects inside the recovered JSON. After tolerant parsing, the output may still be wrong in ways that matter to the business, such as:

- wrong category,
- wrong urgency type,
- made-up policy number,
- missing required fields.

That is exactly why Part B exists: you need schema validation, constrained output, and repair logic, not just a better JSON extractor.

### 3. Which of these would a human reviewer notice in production?

A human reviewer would most notice the outputs that look plausible but are wrong:

- `category_out_of_set`
- `policy_number_invented`
- `missing required fields`
- `urgency_is_string`

Those are the defects that silently poison downstream systems while still looking superficially reasonable.

The parse failures (`markdown_fence`, `prose_around_json`, `malformed_json`) would also be noticed quickly, but usually as a system failure or an empty record rather than as a subtle bad label.

In production, the dangerous cases are the ones that parse cleanly and look convincing while still violating the contract.

---

## 12. Difference between Part A, Part B, and Part C in `extract.py`

The three parts of the Lab 1 extractor are meant to show a progression in how much work is done by the model versus by code.

### Part A — the naive baseline (`v0_naive.py`)

This is the intentionally bad version.

- It sends the ticket to the model.
- It asks the model to return JSON.
- It immediately calls `json.loads()` on the raw response.
- It assumes the model output will always be valid JSON.

This fails because the model often wraps JSON in markdown fences or adds prose. Even when the JSON is recovered, it may still be semantically invalid.

### Part B — schema + validation + repair (`extract.py`, `TicketRecord`)

This is the first real production-style version.

- A Pydantic schema is defined for every allowed field.
- The schema enforces allowed categories, constrained urgency, policy number format, and expected types.
- `structured()` handles JSON mode, tolerant parsing, schema validation, and automatic repair loops.
- The model is still allowed to decide the content fields, but the output is now constrained by a contract.

In short, Part B is about making the model output safe and valid.

### Part C — move deterministic logic out of the model (`extract.py`, `extract_c`)

This is the strongest version of the lab.

- `extract_deterministic()` extracts `policy_number` and `contains_pii` directly from the ticket using code.
- The model is only asked to judge the semantic fields like `category`, `urgency`, `sentiment`, `product`, and `language`.
- `apply_business_rules()` then adds `escalate` in code based on urgency or the presence of the word `ombudsman`.

This is important because things like policy number extraction, PII detection, and escalation logic are deterministic and should not be left to a stochastic model.

### In one sentence

- Part A shows why raw model output is unreliable.
- Part B shows how schema validation and repair make the output usable.
- Part C shows how to push deterministic logic into code so the model only does the genuinely uncertain part.

---
