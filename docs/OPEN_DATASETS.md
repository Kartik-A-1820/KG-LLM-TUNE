# Open Dataset Candidates

Public datasets are useful for format bootstrapping and early smoke tests. They do **not** define Phase 1 success; gates still use Kartik's hand-annotated gold set.

Do not let dataset availability decide the training objective. The target is the complete GraphRAG extraction/indexing flow: entity and relationship extraction, claims/covariates, strict structured output, reference-grounded QA over supplied context, routing labels, and source-grounded summaries/descriptions. Public relation-extraction datasets cover only one slice of that.

## Recommendation for the first pilot

Use **DocRED** first only as a plumbing and format-bootstrap pilot.

- Source: `thunlp/docred` on Hugging Face.
- Licence: Hugging Face tags the dataset as `mit`.
- Shape: document-level relation extraction with entity mentions and relation labels.
- Splits: the Hugging Face card lists `train_annotated` with 3,053 examples, `train_distant` with 101,873 examples, `validation` with 998 examples, and `test` with 1,000 examples.
- Why first: it is small enough for a quick pilot, human annotated, document-level, and licence-cleaner than the alternatives below.
- Limitation: it is not sufficient training data for the GraphRAG model because it does not cover reference-grounded QA, routing, claim extraction, or grounded summaries.

## Do not use yet

**Babelscape/rebel-dataset** is attractive for volume, but it is CC-BY-SA-4.0. That share-alike licence is already called out in `AGENTS.md` and `docs/DATA_STRATEGY.md`; do not include it in the SFT mix until the repo/model licence question is settled.

**TACRED / Re-TACRED** is useful conceptually, but the Hugging Face card marks the licence as `other` and says the underlying corpus is distributed through LDC. Treat it as unavailable until Kartik has licence access and the terms are recorded in `data/README.md`.

**CoNLL04** is small and convenient for a parser smoke test, but the Hugging Face card does not expose a clear licence tag. Use it only after the licence is verified and recorded.

## Pilot Goal

The first open-data pilot should answer a narrow question:

Can the training loop reduce validation loss on public relation-extraction data without violating repo policy?

That is not a Gate 0 or Gate 1 number. It is a pipeline sanity check before spending teacher-distillation money or using any private corpus.
