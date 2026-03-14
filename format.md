# Hackathon: Track B

## AI-Assisted Human-in-the-Loop Workflow for Automated Single-Cell Cloning

**Track B lead**: [Rick Wierenga](https://www.linkedin.com/in/rickwierenga/) / [Will Pierce](https://www.linkedin.com/in/williamcpierce/)

---

## The Problem

Generating a monoclonal cell line (a population of cells descended from a single progenitor) is one of the most fundamental and most tedious tasks in cell biology. It underpins antibody production, CRISPR screening, biologic drug manufacturing, and gene therapy.

In most labs today, the process is still largely manual: a researcher dilutes cells by hand, seeds a plate, checks it under a microscope the next day, and logs results in a spreadsheet. This is slow, operator-dependent, and hard to scale.

The goal is to replace this with an instrumented, software-driven workflow. The robot handles the dilution, the microscope handles the imaging, a model handles first-pass classification, and the human handles decisions that genuinely require human judgment.

---

## The Setting

| Platform | Role |
|---|---|
| **Opentrons OT-2** | Performs all liquid handling steps. A low-cost, open-source liquid handler. |
| **PyLabRobot** | The programmatic interface for the OT-2 in this workflow. Abstracts the instrument from the protocol logic, so the same code can run on different hardware backends. |
| **Cephla Squid** | Images the seeded plate after the OT-2 run. Designed for integration into automated workflows: fully software-controllable and produces structured output that feeds directly into the classification pipeline. |
| **Monomer Culture Monitor** | The handoff point between the ML pipeline and the human reviewer. Purpose-built for cell culture operations, it supports structured per-well data, traceability, and a review interface suited to the QC task. |

---

## Your Challenge

Design, build, and pitch a product for single-cell isolation. The product must:

1. **Dilute and plate** a bead suspension (used as a cell proxy) to maximise single-bead wells in a 96-well plate, while minimising empty wells and avoiding multi-bead wells.
2. **Image the plate** using the Cephla microscope to capture per-well fluorescence images.
3. **Classify each well** using an ML pipeline: e.g., `empty`, `single`, `multiple`, or `uncertain`.
4. **Upload results to Culture Monitor**, including per-well labels and flags for wells requiring review.
5. **Support human review**: a scientist should be able to open Culture Monitor, see flagged wells, and record a final QC decision without touching a spreadsheet.

---

## What You Decide

The problem statement defines inputs, outputs, and integration points. Everything in between is yours:

- How do you characterise and validate your seeding density?
- What features does your classifier use, and how do you evaluate it honestly?
- What threshold determines whether a well is flagged vs. automatically accepted?
- How does a reviewer interact with the Culture Monitor upload?
- What are the failure modes of your pipeline, and how would a user know when to distrust it?

---

## Pitch Format

**30 minutes total: pitch, demo, and judge questions.** How you allocate the time is up to you.

You're building and pitching a product, not just a working pipeline. Your pitch should make clear what the product is, who it's for, and why it's better than the alternative.

Your pitch must include a live demo. What you demo and how is up to you.

---

## How Pitches Are Judged

| Dimension | Weight |
|---|---|
| Technical execution | 25% |
| Evidence quality | 25% |
| Product clarity | 20% |
| Human-in-the-loop design | 15% |
| Communication | 15% |

A technically strong result with weak product thinking scores no better than a credible product with honest, well-reasoned evidence.

---

## Deliverables

Submit before your pitch slot:

- PyLabRobot protocol file(s)
- Image classification pipeline code and any trained model weights
- Culture Monitor upload script or integration code
- A one-page data summary: runs performed, singles counts, pipeline performance metrics
- Pitch slides or equivalent (PDF)

---

*Build something you would trust with real cells.*