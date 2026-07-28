# Using QAMR Skills: A Complete Beginner's Guide

This tutorial shows how to ask an AI coding assistant to use the three Skills
bundled with the Quant Asset Management Research Toolkit (`qamr`). It assumes
no quantitative-investing background. It uses no market data and makes no
investment recommendation.

> **Educational scope.** A Skill helps an AI assistant follow a documented
> workflow. It does not make a prediction, validate an investment decision, or
> turn a historical calculation into financial advice.

## Who this is for

Use this guide if you want to learn how to give a coding assistant a clear,
reproducible request such as: “check my return data”, “calculate a covariance
matrix”, or “compare simple portfolio weights”. You do not need data to start:
the first exercise is a dry run that asks only for a checklist.

By the end, you will be able to:

1. Choose the right QAMR Skill for a question.
2. Give the assistant the assumptions it must not guess.
3. Move from return data to risk estimates, then to transparent portfolio
   weights.
4. Recognise what the Skills deliberately do **not** do.

## 1. What is a Skill?

A Skill is a short, reusable instruction manual for an AI assistant. Instead
of asking a broad question such as “build me a good portfolio”, you point the
assistant to a specific workflow and give it clear inputs. This reduces hidden
assumptions and makes the answer easier to check.

In this repository, the Skill files are source packages; they are not installed
globally. In Codex or Claude Code, ask the assistant to read the relevant local
`SKILL.md` file before it works. For example:

```text
Read skills/qamr-data-boundary/SKILL.md and follow it for this request.
```

That sentence tells the assistant where the rules live. The rest of your prompt
tells it what data and assumptions it may use.

## 2. The three-Skill map

The Skills form a simple sequence. Each one has a different job.

| Step | Skill | Plain-English job | Input | Output |
|---|---|---|---|---|
| 1 | [`qamr-data-boundary`](../skills/qamr-data-boundary/SKILL.md) | Check and label the return data without changing it silently. | A user-provided returns table. | A validated `ResearchDataset` and stated metadata. |
| 2 | [`qamr-covariance-risk`](../skills/qamr-covariance-risk/SKILL.md) | Describe how the assets moved together in the supplied sample. | Validated returns. | A covariance estimate, volatility, correlation, and diagnostics. |
| 3 | [`qamr-portfolio-construction`](../skills/qamr-portfolio-construction/SKILL.md) | Create and compare transparent long-only weight rules. | A covariance estimate. | Weights, portfolio volatility, and risk contributions. |

Think of this as a kitchen workflow: inspect the ingredients first, then measure
them, then choose a recipe. Skipping the first step can make every later result
unreliable.

## 3. Learn the prompt pattern before using data

Good prompts have five small parts:

1. **Skill:** the exact local `SKILL.md` to read.
2. **Input:** a local file or object the assistant may use. If there is no
   input, say so explicitly.
3. **Assumptions:** facts such as return convention, frequency, timezone, and
   any annualisation factor. State them; do not ask the assistant to infer them.
4. **Requested output:** the table, check, explanation, or comparison you want.
5. **Boundary:** reminders such as “do not fetch data” and “do not give trading
   instructions”.

Here is a no-data, safe first prompt. Paste it into an assistant while you are
in the repository root:

```text
Read skills/qamr-data-boundary/SKILL.md and follow it.

I do not have data yet. Do not fetch, invent, or transform any data.
Explain the minimum fields I must provide for a dated return table, then give me
a blank metadata template for: frequency, timezone, return convention, and
missing-data policy. Keep the answer educational and concise.
```

### What you should expect

The answer should ask for **returns**, not prices. It should preserve the dates
and asset labels you provide, and it should not guess a market calendar, a
currency, a timezone, or how missing values should be filled. If it proposes a
portfolio or fetches prices, it has left the Skill's boundary.

## 4. Your first real input: a returns table

When you are ready, provide a CSV or a pandas `DataFrame` containing dated
**returns**. A return is the percentage-like change over one observation
period. For this library, the columns are assets and the rows are observation
dates.

For a first exercise, use synthetic data that you create yourself, or a small
file you are authorised to use. Do not paste credentials, client data, or
confidential market files into an external service.

Before continuing, write down these choices:

| Choice | Example | Why it matters |
|---|---|---|
| Return convention | `SIMPLE` | The meaning of each number must be known. |
| Frequency | `business-day` | Describes the observation interval; it does not automatically annualise risk. |
| Timezone | `UTC` | Makes dates unambiguous. |
| Missing-data policy | `report only` | Prevents hidden filling or deletion of observations. |
| Annualisation factor | `252.0`, if you explicitly choose it | A caller-supplied research assumption used only when annualising. |

`252.0` is only a familiar example for a business-day study. It is not a rule
for every market or dataset. The library intentionally does not infer this
number from a label such as “daily”.

## 5. Skill 1 — validate the data boundary

Replace the path and assumptions below with your own. This prompt asks for a
validation report, not a portfolio.

```text
Read skills/qamr-data-boundary/SKILL.md and follow it.

Use only the dated return table at data/my_returns.csv. Do not fetch data and do
not convert prices to returns. Treat the file as SIMPLE returns with frequency
business-day and timezone UTC. Preserve every date and asset label exactly.
The missing-data policy is report only: do not fill, drop, or realign records
silently.

Adapt the table through PandasAdapter. Report the observation count, date range,
instrument labels, resolved metadata, and every validation warning or error.
Do not estimate covariance or construct a portfolio yet.
```

### Read the result

At this stage, “success” means the assistant can state what it received and
what it checked. It does **not** mean that the assets are attractive. Stop and
fix the input when dates are unordered or duplicated, labels are ambiguous, or
the return convention is unknown.

**Quick exercise:** Change only the missing-data policy in the prompt to the
policy you actually want. Then explain why the assistant should report the
choice instead of deciding it for you.

## 6. Skill 2 — estimate and inspect risk

Only continue after the data-boundary step is clear. Covariance is a compact
way of describing how asset returns varied together in the supplied sample.
Correlation is a related scale from -1 to +1. Neither one predicts the future.

Start with the most transparent estimator: sample covariance.

```text
Read skills/qamr-covariance-risk/SKILL.md and follow it.

Use the validated return dataset from this session. Estimate SampleCovariance
with annualization_factor=252.0. This is my explicit research assumption; do
not infer another factor. Keep the same sample window and labels.

Report the estimator, observation count, annualisation factor, volatility,
correlation, covariance diagnostics, and any PSD policy or repair. Explain in
plain English what each output describes. Do not construct weights, fetch data,
or make predictions.
```

### Choosing an estimator, simply

The Skill can also guide a documented comparison, but it should never choose by
magic:

| Estimator | Beginner interpretation | State explicitly |
|---|---|---|
| Sample covariance | A transparent baseline from the observed sample. | Annualisation factor, if used. |
| EWMA covariance | Gives more weight to more recent observations. | The decay setting. |
| Shrinkage covariance | Can stabilise a noisy sample. | The shrinkage target. |
| Spectral-denoised covariance | Applies an explicit denoising decision. | Rank or effective-observation basis. |

**Quick exercise:** Ask the assistant to compare sample and shrinkage covariance
on the *same* input window. Require it to state the shrinkage target and to
report differences rather than declaring a winner.

## 7. Skill 3 — construct transparent weights

The last Skill uses a covariance estimate to create simple, long-only portfolio
weights. It can compare:

- **Equal weight:** gives every included asset the same weight.
- **Inverse volatility:** gives less weight to assets with higher estimated
  volatility.
- **HRP:** Hierarchical Risk Parity groups related assets before assigning
  weights.
- **HERC:** Hierarchical Equal Risk Contribution follows a related hierarchy
  and risk-allocation approach.

Use this prompt after you have a risk estimate:

```text
Read skills/qamr-portfolio-construction/SKILL.md and follow it.

Use the validated return dataset and the SampleCovariance estimate from this
session. The annualisation factor is 252.0 by my explicit research choice.
Compare equal_weights, inverse_volatility_weights, hrp_weights, and
herc_weights. Use linkage_method="average" for HRP and HERC. There are no
additional constraints.

For each method, report a labelled table of weights and risk contributions,
plus portfolio volatility. State all assumptions and numerical or data-quality
warnings. Do not fetch data, backtest, generate signals, or provide trade
instructions. This is research output, not investment advice.
```

### How to read the table

A weight tells you how the method divided the portfolio under its stated
rules. A risk contribution tells you how much each asset contributes to the
portfolio's estimated volatility. These are different ideas: a small weight can
still contribute meaningful risk when its estimated volatility or relationship
with other assets is high.

Do not conclude that the method with the lowest historical volatility is “best”.
The output depends on the sample, inputs, estimator, and assumptions.

**Quick exercise:** Compare Equal Weight and Inverse Volatility. Identify one
asset whose weight changes. Then ask the assistant to explain the change using
only the reported estimated volatility and labels—not a story about future
returns.

## 8. A complete reusable workflow

Use this checklist every time:

1. Keep source data in a local, authorised location.
2. Ask the data-boundary Skill to validate and label returns.
3. State the risk-estimation assumptions, including any annualisation factor.
4. Ask the covariance-risk Skill to report diagnostics.
5. Ask the portfolio-construction Skill to compare only supported methods.
6. Check that labels match from input to final table.
7. Record assumptions, warnings, and limitations with the output.

If you change the data window, missing-data policy, covariance estimator, or
annualisation factor, say so and rerun the workflow. These are material changes,
not small formatting tweaks.

## 9. Common beginner mistakes

| Mistake | Why it is a problem | Better request |
|---|---|---|
| “Use daily data” with no factor | “Daily” does not tell the library how to annualise. | “Use annualization_factor=252.0; this is my assumption.” |
| Passing prices to a returns workflow | Prices and returns have different meanings. | “This file contains SIMPLE returns; do not convert it.” |
| Letting an assistant repair missing values silently | Hidden changes make results hard to reproduce. | “Report missing values; do not fill or drop records.” |
| Asking for a ‘best’ portfolio | The Skills provide transparent calculations, not a recommendation. | “Compare these four methods and report their assumptions.” |
| Asking the Skills to fetch, backtest, or trade | Those functions are out of scope. | Provide your own returns and keep the request to validation, risk, or weights. |

## 10. Where to go next

After you understand the prompts, run the companion executable notebook:

- [Python Workbook: Quant Portfolio Construction for Complete Beginners](quant_portfolio_construction_for_complete_beginners.ipynb)

The workbook uses synthetic returns to illustrate the calculations in Python.
This guide explains how to give an AI assistant the correct guardrails while it
uses the repository Skills.

### Advanced external reading: tail-sensitive risk

QAMR v1 is deliberately covariance-centred and does **not** implement
Riskfolio-Lib's `RiskFunctions.Kurtosis` or `RiskFunctions.SemiKurtosis`.
These are optional concepts for studying unusually large return observations.
They are **not** QAMR APIs, and they should not be silently substituted for
QAMR volatility or covariance.

The screenshot-style formula is for **SemiKurtosis**, not full Kurtosis. Both
functions take one return series with shape `T x 1`: `T` observations of one
asset or one portfolio. The official Riskfolio-Lib documentation calls their
outputs “Square Root Kurtosis” and “Semi Square Root Kurtosis”.

#### Symbols first

| Symbol | Read it as | Beginner meaning |
|---|---|---|
| $X_t$ | “X at time t” | The return at observation $t$, written as a decimal; for example, $-0.02$ means -2%. |
| $T$ | “number of observations” | How many return observations are in the series. |
| $\mathbb{E}(X_t)$ | “expected X” | The reference average return in the formula. With a finite historical sample, $\bar X$ (the sample average) is a practical estimate to discuss with your analyst. |
| $\min(a,0)$ | “the smaller of a and zero” | Keeps a negative value, but replaces zero or a positive value with zero. |

#### Full Square Root Kurtosis

Riskfolio-Lib documents the full version as:

$$
Kurt(X) =
\left[
\frac{1}{T}\sum_{t=1}^{T}
\left(X_t - \mathbb{E}(X_t)\right)^4
\right]^{1/2}.
$$

For every observation, this formula measures the distance from the reference
average, raises the distance to the fourth power, averages those values, and
then takes a square root. The fourth power makes large deviations matter much
more: a deviation that is twice as large contributes $2^4 = 16$ times as much
before averaging.

#### Semi Square Root Kurtosis — the formula in the screenshot

Riskfolio-Lib documents the downside-only version as:

$$
SemiKurt(X) =
\left[
\frac{1}{T}\sum_{t=1}^{T}
\min\!\left(X_t - \mathbb{E}(X_t), 0\right)^4
\right]^{1/2}.
$$

Read it from left to right:

1. Find each return's distance from the reference average.
2. Keep only negative distances—returns below that reference average. Positive
   distances become zero because of $\min(\cdot,0)$.
3. Raise each retained downside distance to the fourth power, so unusually deep
   negative observations receive extra weight.
4. Average across all $T$ observations, including the zero contributions.
5. Take the square root.

This is why SemiKurtosis is a **downside-sensitive** fourth-moment measure: it
ignores above-reference deviations but magnifies below-reference deviations.

#### Tiny worked example

Suppose three decimal returns are $[-0.02, 0.00, 0.02]$, so the sample average
is $\bar X = 0$. The only below-average return is $-0.02$.

$$
SemiKurt(X) =
\left[\frac{(-0.02)^4 + 0^4 + 0^4}{3}\right]^{1/2}
= \left[\frac{0.00000016}{3}\right]^{1/2}
\approx 0.00023094.
$$

The $+2\%$ observation is above the average, so SemiKurtosis turns its
contribution into zero. Full Kurtosis would include both the $-2\%$ and the
$+2\%$ deviations. The numerical result is a description of this tiny sample,
not a prediction of future losses.

#### Use it carefully

- This is Riskfolio-Lib's fourth-moment-derived measure, not automatically the
  same convention as the “excess kurtosis” number reported by another library.
- Fourth powers make the result especially sensitive to extreme observations
  and to small samples. Always report the sample window and observation count.
- Keep the return convention explicit and do not mix prices with returns.
- Use the official references for the exact external API:
  [Kurtosis](https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.Kurtosis-returns)
  and
  [SemiKurtosis](https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.SemiKurtosis-returns).

If you ask an assistant to explain either measure, keep the request scoped:

```text
Use the Riskfolio-Lib Kurtosis and SemiKurtosis documentation only as external
reading. Explain the equation term by term for my provided return series. Do
not claim that qamr implements these functions, do not fetch data, and do not
turn the statistic into an investment recommendation or a forecast.
```

## Content note

Gemini 3.5 Flash assisted with brainstorming the beginner-learning outline.
The repository-specific claims, supported methods, prompt boundaries, and code
identifiers in this guide were checked against the three local QAMR Skill files
before publication.
