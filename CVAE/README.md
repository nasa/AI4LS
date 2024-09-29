# Sarah Golts' SLSTP Project Repo

Assessing spaceflight data generation capabilities of conditional variational autoencoders (vanilla and gene module annotated).

## Getting Started

Set up your own environment (venv or Conda) and clone the repo:

```sh
git clone https://github.com/sgolts/slstp_cvae.git
```

Make sure Python 3.x is being used. Install required dependencies:

```sh
pip install -r requirements.txt
```

## Hyperparameter Selection

Due to GitHub file size limits, raw and integrated data cannot be provided in the repo (I can email it).

If integrated data has not been emailed, run `data.py` on raw .h5 files. No changes to code needed for this -- raw files are the .h5 files from CellRange pipeline, just place in a `\data` directory in the project directory. Processed and integrated files will be saved to `\data` directory.

Run `hyperparam_sweep.py`. By default performs sweep for both Vanilla CVAE and GMA CVAE using 5-fold cross-validation. If one model's parameters are not needed, can comment out corresponding section.

## Training Models

Input selected hyperparameters into `train_models.py`. Run for appropriate number of epochs. Pre-trained models are stored in the `\trained_models` directory.

## Generating Data

Use `generate_data.py`. Set variable `model_for_gen` to desired model to use for generation:

```sh
# To generate using the Vanilla CVAE:
model_for_gen = 'vanilla'

# To generate using the GMA CVAE:
model_for_gen = 'write in anything else!'
```

To modify conditions for generating samples edit the conditions tensor on line 51 using the following guidelines:

['Strain' = 0 - 2, 'Sex' = 0 (female), 'Age at Launch' = int of weeks, 'Duration' = int of days, 'Flight' = 0 for ground, 1 for flight].

For example, for a mouse of strain 1 (insert strain name), female, 14 weeks old at launch, that went on a 28 day spaceflight mission, the line would look like:

```sh
conditions = torch.tensor([1, 0, 14, 28, 1], dtype=torch.float32)
```

To change number of samples generated, modify the `num_samples` parameter on line 52. For example, to generate 1000 samples, the line would be:

```sh
generated_data = model.generate(conditions, num_samples=1000).numpy()
```
