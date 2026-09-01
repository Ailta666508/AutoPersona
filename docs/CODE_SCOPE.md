# Code and Research Alignment

This note distinguishes the broader AutoPersona research workflow from the software included in this public repository.

## Implemented here

| Research concept | Reference implementation |
| :--- | :--- |
| Trajectory, Workspace, and Persona memories | `models.py`, `refinement.py` |
| Per-user persistent memory bank | `store.py` |
| Similarity retrieval across memory layers | `retrieval.py` |
| Memory add/update/delete resolution | `extraction.py` |
| PersonaAgent retrieval and clarify/final decision | `persona_agent.py` |
| Dependency-aware personalized execution | `execution.py` |
| Persona2Web record conversion | `adapters/persona2web.py` |
| Runtime counters | `metrics.py` |

## Part of the research, not included

- SFT data construction and model training;
- the GRPO reward implementation and optimization loop;
- trained-model integration;
- benchmark datasets and redistribution-controlled assets;
- Persona2Web, PersonaMem-v2, and PI-Bench evaluation runners;
- external executor and judge-model integrations;
- experiment configurations, checkpoints, raw predictions, or statistical analysis;
- scripts that regenerate unpublished tables or figures;
- the research manuscript and submission metadata.

The 11-file core package is therefore suitable for inspecting the runtime abstractions and testing memory-aware control flow. It is not an artifact-complete scientific reproduction package.

## Excluded vendor source

The original local folder contained an unpacked copy of `verl 0.7.0`, an independent reinforcement-learning framework with its own authors, history, and Apache-2.0 license. None of the AutoPersona core modules imports that local tree. Copying the entire vendor project into this repository would obscure code ownership, inflate language statistics, and make the repository appear to claim third-party work. It is intentionally excluded.

If training code is added later, declare the supported upstream `verl` version as a dependency or submodule, preserve its license and notices, and document all AutoPersona-specific changes separately.

## Curated release preparation

The original core files are copied byte-for-byte under `autopersona_memory/`. New release material consists of package metadata, English documentation, tests, a deterministic example, ignore rules, continuous integration, and release-verification scripts. The original hashes are recorded in `source-manifest.json`.

## Public release boundary

This repository intentionally excludes:

1. manuscript and submission files;
2. venue names and submission identifiers;
3. unpublished numerical results and result tables;
4. benchmark data, model checkpoints, raw predictions, and private logs;
5. third-party source that is not part of the AutoPersona implementation.

Future releases should preserve this boundary until an appropriate public citation and rights-cleared research artifact are available.
