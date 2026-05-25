# Clinical Contextual Redundancy - Thesis project

This is the full pipeline for my experiment, estimating mutual information in clinical spontaneous speech.

This repo applies the lightning-hydra repo structure and regression modules originally applied in:

https://github.com/Chief-Buka/contextual-redundancy

The repo is configured for using 4 distinct datasets (3 corpora from Discourse and 1 from DAIC-WOZ), why we have dataset specific notebooks (TANG, UWO, TOPSY and DAIC).

For the pipeline breakdown, we'll just use DAIC-WOZ as an example.

This assumes that you store your data in per-speaker folders, containing full audio files and transcriptions, under:

```text
./Data/YOUR_DATASET/SPEAKER_FOLDER(S)
```

If working with multiple datasets, repeat steps 2–5 per dataset.


# 1. Create environment for audio processing and prosodic feature extraction

```bash
conda create -n extraction python=3.10

conda activate extraction

cd path/to/this/repo

pip install -r audio_reqs.txt

conda install -c conda-forge montreal-forced-aligner
```

Then we can run the notebooks — adjust paths to data as needed.


# 2. Chunk audio files and transcripts into audio chunks of 5–25 seconds

Notebook:

```text
audio_segmenting.ipynb
```

This notebook will use timestamps in the transcript file (`.cha`, `.txt` or `.csv`) to create audio chunks from arguments assigned at the start of the notebook.

Chunks will be stored with cleaned transcripts in speaker specific folders with an utteranceID suffix:

```text
speakerID_utteranceID.txt
speakerID_utteranceID.wav
```

This also creates a log file over utterances not used and the reason why (e.g. too long, few words).

## 2.5 Optional: Whisper retranscription

If transcriptions are not good enough, run:

```text
/Data/whisper_transcription.ipynb
```

on the chunks.

The notebook is configured to include disfluencies as much as possible for better alignment in the next step.


# 3. Forced align the chunks using MFA

Make sure to download corresponding MFA language models and dictionaries and place these in the full `audio_chunked` root repo.

MFA documentation:

https://montreal-forced-aligner.readthedocs.io/en/latest/reference/language_modeling/index.html

```bash
cd to/chunked/audio/dir

mfa align CORPUS_DIRECTORY DICTIONARY_PATH ACOUSTIC_MODEL_PATH OUTPUT_DIRECTORY
```

If you have stored language models and dictionaries as instructed, this could for instance be:

```bash
mfa align dataset_name english_us_mfa.dict english_mfa.zip dataset_name_output
```

After alignment, move TextGrids into corresponding utterance audio folders.


# 4. Create train/test/validation splits

Run:

```text
Data/DAIC-WOZ/create_splits_DAIC.ipynb
```

on the assigned data repo.

This is done based on:
- speakers
- mental health
- overall audio duration

The notebook currently includes:
- CDS
- PANSS-8/10
- PHQ-8

for labeling psychosis and depression, but more can be added.


# 5. Sort data into correct structure

Run:

```text
Data/DAIC-WOZ/organize_data.ipynb
```

based on splits created.


# 6. Run prosodic feature extraction

Run:

```text
prosodic_feature_extraction.ipynb
```

This requires:
- structure created by previous steps
- a metadata file about speakers

This will create a CSV with all extracted feature values in each column (optionally along with metadata).

This step can take several hours.


# 7. Concatenate datasets and generate PKL/YAML configs

Once prosodic feature extraction has been performed for all datasets, run:

```text
concat_dfs_to_pkl_and_yaml.ipynb
```

This will:

1. Concatenate all distinct dataset pandas dataframes
2. Create `.pkl` dataset files for every feature, based on train/test/val splits
3. Create additional `test.pkl` files based on mental health status encoded in `utterance_id` in steps 4 & 5
4. Create config YAML files in correct contextual-redundancy folders, allowing for model regression


# 8. Create environment for training

```bash
conda create -n prosody python=3.10

conda activate prosody

cd path/to/this/contextual-redundancy-repo

pip install -r training_reqs.txt
```


# 9. Model regression

## Initial trial run

To see if everything is working, you can do an initial trial run for a feature (e.g. Duration):

```bash
TOKENIZERS_PARALLELISM=false python -m src.train experiment=emnlp/finetuning/duration_finetuning trainer.max_epochs=1 trainer.accumulate_grad_batches=1 +trainer.limit_train_batches=5 +trainer.limit_val_batches=2 +trainer.limit_test_batches=2
```

## Full training

```bash
TOKENIZERS_PARALLELISM=false python -m src.train experiment=emnlp/finetuning/duration_finetuning
```

Then run for every feature.

## Testing individual test splits

For example depressive speech:

```bash
TOKENIZERS_PARALLELISM=false python -m src.train experiment=emnlp/finetuning/duration_finetuning train=False test=True ckpt_path="/logs/train/runs/FIND/MODEL/CHECKPOINT.ckpt" data.test_file=test_depression.pkl
```

This loads the best model checkpoint from training and performs testing only.


# 10. Compute mutual information

Run:

```text
compute_plot_MI.ipynb
```

This notebook will:

1. Estimate unconditional entropy (NLL) matching the distributional family for every feature used during regression
2. Estimate conditional entropy based on the losses logged by the models during training and testing for each feature
3. Compute mutual information per feature as:

```text
MI = unconditional entropy - conditional entropy
```

4. Visualize the time-scale of mutual information in heatmaps


# 11. Done

You can use:

```text
data_summary.ipynb
```

for statistics of the data used.


# Changes to the original repository 
https://github.com/Chief-Buka/contextual-redundancy

## Added / modified functionality

- New plotting notebook computing differential entropy using parametric distributions
- New notebook for estimating differential unconditional entropy using KDE
  - Use this for more robustness
  - Not applied in the current repo setup
  - Can take several hours
- Added feature extraction notebook utilizing modules in the original repo for new data
- Added a new `custom_pickle_datamodule`, allowing new extracted data to be passed to model regression
- Added notebook for generating `.pkl` files using generated dataframes
- This notebook also generates:
  - new `finetuning.yaml` files in:
    ```text
    configs/experiment/emnlp/finetuning
    ```
  - new `_datamodule.yaml` files in:
    ```text
    configs/data/
    ```
- All `TokenTaggingRegressor` files:
  - changed AdamW imports to `torch.optim`
  - commented out `memory_profiler` imports
- Added settings generating 4 context windows per utterance to better utilize limited data
- `torch_metrics.py` and `torch_metrics_cm.py`
  - updated `MaskedPearsonCorrCoeff`
  - removed dependency on local helper functions due to dependency conflicts
- Updated requirements:
  - split into:
    ```text
    audio_reqs.txt
    training_reqs.txt
    ```
  - allowing use of 2 separate environments
- Added data summary notebook for producing statistics on the data
