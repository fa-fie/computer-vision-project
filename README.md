# cv-project

Traffic sign classification on GTSRB with adversarial robustness experiments.

## Structure

```
architectures.py           AlexNet (43 classes)
utils.py                   dataset loading helpers
evaluate.py                evaluation on attack test sets
plotting.py                loss/accuracy plots
run_experiments.py         inference and visualisation

attack_classifier.py       binary CNN: clean vs attacked
attack_detection_dataset.py  datasets for the classifier and E2E training
pipeline.py                full pipeline + training loops

physical_adv_attack/
  generator.py             generates attacked images (4 attack types)
  run.py                   run the generator
  config.yaml              generator config

ComputerVision_Groupproject.ipynb   main notebook (use this in Colab)
docs/adversarial_detection_pipeline.md   notes on the new pipeline
```

## Setup

```bash
pip install -r requirements.txt

# generate attacked images (set paths in config.yaml first)
python physical_adv_attack/run.py
```

Open `ComputerVision_Groupproject.ipynb` in Colab.
- Sections 1–3: original training and adversarial training
- Section 4: attack classifier and E2E pipeline

## Attacks

| Name | What it does |
|------|-------------|
| occlusion | coloured stickers on the sign |
| shadow | triangular shadow across the sign |
| noise_blur | Gaussian noise + blur inside the sign area |
| graffiti | text drawn on the sign |
