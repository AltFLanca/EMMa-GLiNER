
# EMMa-GLiNER

Explore generic product type extraction from EMMa product titles using [GLiNER](https://github.com/urchade/GLiNER), a named entity recognition model. The extracted product types can be used as potential text prompts for product segmentation. For example:

```text
Title: "Canon EF 100mm f/2.8L IS USM Macro Lens for Canon Digital SLR Cameras"
Generic product type: "Macro Lens"
```

Dataset access is handled through [EMMa-Wrapper](https://github.com/AltFLanca/EMMa-Wrapper).

## Setup

Install the project dependencies using [uv](https://docs.astral.sh/uv/). From the repository root, run:

```bash
uv sync --locked
```

A prepared EMMa dataset is required. The dataset and GLiNER model weights are not included in this repository.

## Repository Structure

- `src/test_inference.ipynb` - Notebook for testing generic product type extraction on EMMa product titles.
- `src/evaluation_test.py` - Evaluates extraction performance against a fixed set of annotated product titles.

## Evaluate Extraction

The [evaluation_test.py](src/evaluation_test.py) script evaluates GLiNER extraction performance against a fixed set of 64 manually annotated product titles, each associated with one or more acceptable product type aliases.

Run evaluation with:

```bash
uv run src/evaluation_test.py \
  --model-path /path/to/GLiNER
```

Predictions are compared against the annotated aliases using text matching and similarity-based metrics. The script prints summary results for the evaluated configurations.

**Note:** The current implementation is intended for exploratory prompt extraction and evaluation. It does not provide a dataset-wide augmentation generation pipeline.