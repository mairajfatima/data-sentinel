# 🛡️ Data Sentinel

**👉 Try it live: https://mairajfatima.github.io/data-sentinel/**
*(or go straight to the app: https://data-sentinel-hbquh73gibgejxgugq2raq.streamlit.app)*

An all-in-one CSV data validation tool — upload any CSV and get structural checks, rule checks, statistical analysis, and ML-based anomaly detection, all in one place, in seconds.

---

## Why Data Sentinel

Before trusting a CSV — before building a dashboard, report, or model on top of it — someone has to check it's actually clean. Today that means jumping between multiple tools and tabs: one tool to check nulls and duplicates, another to eyeball statistics, a notebook to hunt for outliers, maybe a separate chatbot tab to ask questions about what you found.

**Data Sentinel puts all of that in one product, one upload, one report:**

- No switching between a spreadsheet, a notebook, and a chatbot tab
- No manual eyeballing for nulls, duplicates, or weird values
- No writing one-off scripts every time a new file needs checking
- Works on **any CSV, any domain** — sales, HR, surveys, inventory, anything — because it adapts to each file's columns automatically, with zero configuration

It's the difference between *assembling* your own validation workflow from five tools, and getting the full picture in one place the moment you upload a file.

---

## What It Does

- ✅ Structural & rule-based checks — nulls, duplicates, negative values, ranges
- 🔤 Pattern checks — whitespace issues, basic email-format validation
- 📊 Statistical analysis — mean, median, std, row-count sanity checks
- 🤖 **ML-based anomaly detection** — StandardScaler + PCA (visualization) + Isolation Forest + K-Means working together to flag unusual rows and explain *why* each one was flagged, in plain English
- 📈 Auto-suggested KPIs based on the columns actually present in your file
- 💬 Built-in chatbot — ask questions about your specific report, grounded strictly in your data's results
- 📄 One-click PDF export of the full report

All in one upload. No tab-switching.

---

## What's Coming Next

This is an active, evolving product. Planned next:

- Configurable business-logic rules (e.g. domain-specific validation beyond generic type checks)
- Cross-file reconciliation (compare an upload against a "source of truth" file)
- Persistent run history / audit log across sessions
- More chart types and deeper chatbot visualization support

---

## Try It

**Live app:** https://data-sentinel-hbquh73gibgejxgugq2raq.streamlit.app
**Landing page:** https://mairajfatima.github.io/data-sentinel/

Built by [Mairaj Fatima](https://github.com/mairajfatima).
