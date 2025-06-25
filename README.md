# 🧬 Biomedical Label Annotation with GPT-4o-mini: Base, Fine-Tuned, and RAG Models

This repository contains the code and evaluation scripts for the paper:

**"Integrating Retrieval-Augmented Generation with the BioPortal Annotator for Biological Sample Annotation"**

## 📦 Overview

This project explores three strategies for annotating biological sample labels using ontology terms:

- **Base GPT-4o-mini** model (zero-shot)
- **Fine-tuned GPT-4o-mini** model
- **Retrieval-Augmented Generation (RAG)** using BioPortal

The goal is to assess the performance of each method and determine whether RAG offers a cost-effective, high-precision alternative to fine-tuning, aligned with FAIR data principles.

---

## 📚 Dataset

We reuse a manually curated dataset of **6,264 biomedical labels** from 27 public databases, annotated with four ontologies:

- **CLO** (Cell Line Ontology)
- **CL** (Cell Ontology)
- **UBERON** (Uber-anatomy Ontology)
- **BTO** (BRENDA Tissue Ontology)

A test subset of **1,880 labels** was used for evaluation, categorized into:

| Type               | Labels |
|--------------------|--------|
| Cell Lines (CL)    | 918    |
| Cell Types (CT)    | 696    |
| Anatomical (A)     | 208    |
| Unclassified       | 58     |

Gold standard annotations are available in [`biosamples.tsv`]() and [`mappings_test.tsv`]().

---

## ⚙️ Methods

### GPT-4o-mini Configurations

All models were accessed via the OpenAI API:

- **Base**: zero-shot prompt-based inference.
- **Fine-tuned**: trained on labeled data for this task.
- **RAG**: integrates results from BioPortal Annotator at runtime.

Prompts were structured to define task scope, output format, and constraints (e.g., one valid identifier per label, no extra text).

### 🔎 RAG + BioPortal Annotator

RAG uses the **BioPortal Annotator** to retrieve candidate ontology terms for each label. These candidates are included in the prompt as context, and GPT-4o-mini selects the most appropriate class.

This approach enhances semantic matching while maintaining interpretability and cost-efficiency.

---

## 📊 Evaluation

### Metrics Used

- **Precision**
- **Recall**
- **F1-score**
- **Accuracy**
- **Perfect Match Ratio**: proportion of labels with all expected ontology identifiers correctly predicted.

### Evaluation Strategy

- Predictions were matched to gold standard annotations using exact and semantic matching (synonyms, subclasses, etc.).
- Errors were qualitatively reviewed to distinguish between true errors and acceptable variations.

---

## 📁 Repository Structure

