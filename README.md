# MMA3001-Frozen-Meat-Quality
# MMA3001 — Frozen Meat Packaging Quality Detection

## Project Overview

This project investigates automated defect detection in frozen-meat packaging using computer vision and deep learning.

The objective is to identify and localise visible packaging defects in RGB images using object-detection models.

Two approaches are investigated:

* YOLO11n
* RF-DETR

Their detection performance is compared using precision, recall, F1-score and bounding-box localisation metrics.

## Engineering Problem

Manual inspection of frozen-meat packaging can be time-consuming and subject to inconsistent assessment.

Automated image-based inspection offers a potential method for identifying visible defects and supporting packaging quality-control decisions.

The computational workflow processes RGB images and predicts defect classes, bounding-box coordinates and confidence scores.

## Dataset

The dataset contains 1,955 RGB images divided into:

| Split      | Images |
| ---------- | -----: |
| Training   |  1,764 |
| Validation |    115 |
| Testing    |     76 |

Five defect classes are considered:

1. Loose meat
2. Packaging error
3. Twisted meat
4. Unsealed
5. Wrinkle

The images use YOLO-format bounding-box annotations.

The dataset is not included in this repository. Users must provide the dataset separately to reproduce training and evaluation.

## Computational Methods

### YOLO11n

YOLO11n is used as a lightweight object-detection approach for identifying and localising packaging defects.

### RF-DETR

RF-DETR is investigated as an alternative object-detection architecture.

Both approaches are evaluated using labelled images to compare detection accuracy and localisation performance.

## Repository Contents

* `MMA3001_Frozen_Meat_Quality.ipynb` — Google Colab notebook containing the experimental workflow.
* `src/evaluation.py` — Reusable evaluation-metric calculations.
* `tests/test_evaluation.py` — Automated unit tests.
* `docs/index.html` — HTML documentation generated using pdoc.
* `test_report.html` — Automated pytest report.
* `requirements.txt` — Python dependencies.
* `LICENSE` — MIT licensing agreement.

## Installation

The experimental notebook was developed using Google Colab.

To install the project dependencies, run:

```bash
pip install -r requirements.txt
```

## Running the Project

1. Open `MMA3001_Frozen_Meat_Quality.ipynb` in Google Colab.
2. Install the required dependencies.
3. Provide the RGB image dataset and YOLO-format annotations.
4. Update the notebook's dataset and output paths to match your environment.
5. Run the relevant preprocessing, training or evaluation cells in sequence.
6. Review the generated model predictions and performance metrics.

Model training requires substantial computational resources. A GPU-enabled environment is recommended.

The original notebook contains environment-specific paths and may require configuration before execution.

## Evaluation Metrics

The project evaluates object-detection performance using:

* Precision
* Recall
* F1-score
* Intersection over Union (IoU)
* Mean Average Precision (mAP)

The reusable evaluation module calculates precision, recall and F1-score from true-positive, false-positive and false-negative counts.

## Automated Testing

Unit tests are implemented using pytest.

To run the tests from the repository root:

```bash
python -m pytest tests -v
```

To generate a new HTML test report:

```bash
python -m pytest tests -v --html=test_report.html --self-contained-html
```

The tests verify the evaluation utility, including perfect detection, zero detections, known metric values and invalid inputs.

These unit tests do not constitute end-to-end validation of the complete model-training workflow.

## HTML Documentation

Source-code documentation was generated using pdoc.

Open `docs/index.html` in a web browser to access the generated documentation.

The HTML documentation covers the reusable evaluation module. The notebook contains the remaining experimental implementation.

## Limitations

The project has several limitations:

* Class imbalance affects the availability of representative training examples.
* The wrinkle class has no labelled validation or test examples.
* Bounding-box localisation remains challenging for some defect categories.
* The available test dataset is relatively small.
* Similar-looking images may affect the independence of evaluation samples.
* Reproducing the full training workflow requires external data and computational resources.

## License

This repository is distributed under the MIT License. See the `LICENSE` file for details.
