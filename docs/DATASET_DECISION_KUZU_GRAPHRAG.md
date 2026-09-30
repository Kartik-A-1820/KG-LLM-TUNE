# Dataset Decision: Local Kuzu GraphRAG

This project is not training a generic relation extractor. It is training
SmolLM2-360M-Instruct to be the local LLM worker inside a fully local GraphRAG
stack backed by a property graph, with Kuzu as the target graph database.

The dataset mix is chosen by pipeline capability, not by benchmark popularity.

## Target Runtime

- LLM: `HuggingFaceTB/SmolLM2-360M-Instruct`
- Graph DB: Kuzu, an embedded property graph database that implements Cypher
  (`https://github.com/kuzudb/kuzu`)
- Retrieval: hybrid local retrieval across vector search, keyword/full-text
  search, graph neighborhood expansion, and optional Cypher query generation
- Embedder: selected separately through the embedding A/B plan

## Capabilities The Model Needs

### 1. Graph Construction From Text

Given a source chunk, emit a property-graph-ready JSON object:

```json
{
  "entities": [
    {
      "id": "e0",
      "name": "surface form",
      "canonical_name": "canonical form",
      "type": "person|organization|product|place|concept|...",
      "description": "source-grounded span or near-span",
      "source_span": {"start": 0, "end": 10}
    }
  ],
  "relations": [
    {
      "head": "e0",
      "tail": "e1",
      "type": "RELATION_TYPE",
      "description": "source-grounded span or near-span",
      "evidence": "verbatim supporting text"
    }
  ],
  "claims": []
}
```

Training data for this capability needs entity extraction, entity typing,
relation extraction, claim/event extraction, grounded descriptions, and evidence
spans.

### 2. Graph Hygiene

The model must help keep the graph usable, not merely large:

- canonicalize aliases inside a chunk
- identify possible cross-chunk duplicates
- emit stable entity types
- avoid dangling relation endpoints
- keep descriptions grounded in the source text
- avoid relation labels that are synonyms for the same edge type

This requires data that teaches normalization and evidence, not only triples.

### 3. Kuzu/Cypher Query Planning

For graph-only or graph-heavy questions, the model should translate user intent
and a supplied schema into Kuzu-compatible Cypher.

This is separate from extraction. Text-to-Cypher data teaches how to query a
finished graph; it does not teach how to build one.

### 4. Retrieval Routing

Given a user question, the model should choose among:

- vector-only retrieval
- keyword/full-text retrieval
- graph neighborhood expansion
- Cypher query
- hybrid retrieval with reciprocal-rank or weighted fusion
- answer directly from supplied context

This requires synthetic and/or hand-labelled routing data, because public
datasets rarely label retrieval strategy.

### 5. Grounded Context QA

Given retrieved chunks, graph facts, paths, and source snippets, the model should
answer only from supplied evidence and cite which evidence supports each answer.

This is local answer drafting over retrieved context. Gate 3 still decides
whether the local graph supports answers close enough to the teacher graph.

### 6. Source-Grounded Summaries

The model needs summaries for chunks, entities, relations, and neighborhoods.
These must be source-grounded, because fabricated descriptions poison graph
retrieval.

## Dataset Mix Decision

### Candidate Components

**DocRED**

Use for document-level entity and relation extraction bootstrapping. It is useful
because relations can cross sentence boundaries. It is not enough by itself.

**IEPile: hold out of the default mix.**

The Hugging Face dataset card labels IEPile CC BY-NC-SA 4.0 and states that
source datasets with stricter terms retain those terms. That is not a clean fit
for an organization-ready training mix. Reconsider only after legal review of
the license and constituent datasets
(`https://huggingface.co/datasets/zjunlp/iepile`).

**Neo4j Text2Cypher 2024**

Use for Cypher query generation and query-planning practice, then normalize the
outputs to the Kuzu-supported Cypher subset. The dataset card records
`question`, `schema`, and `cypher` fields, Apache-2.0 license, 39,554 train
examples, and 4,833 test examples
(`https://huggingface.co/datasets/neo4j/text2cypher-2024v1/raw/main/README.md`).
The dataset combines many upstream sources; their individual terms have not
been independently reviewed, so this remains a diagnostic pilot source until
that review is complete.

### Use Only After Licence Review

**REBEL / RED-style relation extraction**

Good for open triple extraction volume, but license/share-alike implications must
be resolved before use.

**SciERC / TACRED / Re-TACRED / CoNLL04**

Useful for typed entity-relation extraction, but each dataset needs license and
domain-fit review before inclusion.

**SynthCypher**

Potentially useful for broader Text2Cypher, but do not include until its
non-commercial/share-alike constraints are reviewed.

### Build Ourselves

These are too project-specific to trust a public dataset alone:

- Kuzu property-graph JSON extraction examples
- Kuzu schema generation from extracted entities/relations
- Kuzu Cypher compatibility fixes
- retrieval-router labels
- graph-neighborhood summarization
- source-grounded entity and relation descriptions
- answer-from-hybrid-context examples
- negative examples for hallucinated descriptions and dangling edges

Teacher-generated data can cover these, but only after the policy blockers about
corpus export and teacher-output terms are closed.

## Training Mixture

Start with this rough mixture by example count, then tune after the first
diagnostic runs:

| Capability | Share | Primary source |
| --- | ---: | --- |
| Property-graph extraction JSON | 35% | DocRED for bootstrap; license-reviewed RE/OpenIE or teacher-generated Kuzu JSON for production mix |
| Entity normalization and graph hygiene | 10% | synthetic aliases, teacher-generated duplicate/canonicalization examples |
| Source-grounded descriptions and summaries | 15% | teacher-generated from chunks, span/near-span summaries |
| Grounded QA over supplied context | 15% | context QA datasets plus teacher-generated corpus examples |
| Retrieval routing | 10% | synthetic/hand-labelled routing decisions |
| Kuzu/Cypher generation | 15% | Neo4j Text2Cypher 2024 normalized to Kuzu-compatible Cypher |

These percentages are starting design weights, not measured performance numbers.
They must not be reported as results.

## What Not To Do

- Do not train only on DocRED and call it GraphRAG-ready.
- Do not train only on Text2Cypher; that teaches querying, not graph building.
- Do not use generic summarization datasets unless outputs are source-grounded.
- Do not train on unsupported free-form relation labels without mapping them into
  a controlled schema.
- Do not include any private corpus text in Kaggle or public commits until the
  policy blockers are answered.

## Local Pilot Prepared

`scripts/prepare_kuzu_pilot.py` builds a reproducible 5,000-row local diagnostic
mix from DocRED extraction rows and Neo4j Text2Cypher rows, with disjoint
validation inputs and source hashes in its ignored local manifest. The current
mix exercises only graph extraction and Cypher generation. It is not the full
GraphRAG training mix and its validation loss cannot establish semantic quality
or Kuzu query compatibility. No private corpus is used.
