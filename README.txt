This is the full pipeline for my experiment, estimating mutual information in clinical spontaneous speech.
This repo applies the lightning-hydra repo structure and regression modules originally applied in https://github.com/Chief-Buka/contextual-redundancy

The repo is configured for using 4 distinct datasets (3 corpora from Discourse and 1 from DAIC-WOZ), why we have dataset specific notebooks (TANG, UWO, TOPSY and DAIC)
For the pipeline breakdown, we'll just use DAIC-WOZ as an example.

This assumes that you store your data in per-speaker folders, containing full audio files and transcriptions, under ./Data/YOUR_DATASET/SPEAKER_FOLDER(S)

If working with multiple datasets, repeat steps 2-5 per dataset.

1. Create env for audio processing and prosodic feature exctraction:

	conda create -n extraction python=3.10
	
	conda activate extraction

	conda cd path/to/this/repo

	pip install -r audio_reqs.txt

	conda install -c conda-forge montreal-forced-aligner

Then we can run the notebooks - adjust paths to data as you need:

2. chunk audio files and transcript into audio chunks of 5-25 secs length. (audio_segmenting.ipynb)
	This notebook will use timestamps in the transcript file (.cha, .txt or .csv) to create audio chunks from arguments assigned at the start of the notebook.
	Chunks will be stored with cleaned transscripts in speaker specific folders with a utteranceID suffix: speakerID_utteranceID.txt / .wav
	This also creates a log file over utterances not used and the reason why (e.g. too long, few words)

   (2.5) If transcriptions are not good enough, run /Data/whisper_transcription.ipynb on the chunks. 
	The Notebook is configured to include disfluencies as much as possible for better alignment in next step.

3. Forced align the chunks using MFA. (Make sure to download corresponding MFA language models and dicts and place these in the full audio_chunked root repo)
   (https://montreal-forced-aligner.readthedocs.io/en/latest/reference/language_modeling/index.html)
	
	$ bash
		
	cd to/chunked/audio/dir

	mfa align CORPUS_DIRECTORY DICTIONARY_PATH ACOUSTIC_MODEL_PATH OUTPUT_DIRECTORY

	
    So if you have stored language models and dicts as intructed this could for instance be

	
	mfa align dataset_name english_us_mfa.dict english_mfa.zip dataset_name_output


    After alignment, move textgrids into corresponding utterance audio folders.  	

3. Create train/test/val splits, by running Data/DAIC-WOZ/create_splits_DAIC.ipynb on assigned data repo. This is done based on speakers, mental health and overall audio duration. 
   	The notebook currently includes CDS, PANSS-8/10 and PHQ-8 for labeling psychosis and depression, but more can be added.
 
4. Sort data into correct structure (Data/DAIC-WOZ/organize_data.ipynb) based on splits created

5. Run prosodic_feature_extraction.ipynb to perform feature extraction: this requires structure created by previous steps aswell as a metadata file about speakers. 
   This will create a csv with all feature values extracted in each column (optionally along with metadata). It will take some hours.

6. Once prosodic feature extraction has been performed for all datasets, run concat_dfs_to_pkl_and_yaml.ipynb. This will:
	i Concatenate all distinc dataset pandas dataframes
	ii. Create .pkl dataset files for every feature, based on train/test/val splits.
	iii. Create additional test.pkl files based on mental health status encoded in utterance_id in steps 3 & 4. 
	iiii. Create config yaml files in correct contextual-redundancy folders, allowing for model regression.

7. Create new env for training:

	conda create -n prosody python=310
	
	conda activate prosody

	conda cd path/to/this/contextual-redundancy-repo

	pip install -r training_reqs.txt

8. Model regression! To see if everything is working, you can do an initial trial run for a feature (e.g. Duration):

	$ bash
	TOKENIZERS_PARALLELISM=false python -m src.train experiment=emnlp/finetuning/duration_finetuning trainer.max_epochs=1 trainer.accumulate_grad_batches=1 +trainer.limit_train_batches=5 +trainer.limit_val_batches=2 +trainer.limit_test_batches=2


   For training run:

	$ bash
	TOKENIZERS_PARALLELISM=false python -m src.train experiment=emnlp/finetuning/duration_finetuning

   Then run for every feature.


   For TESTING individual test-splits. e.g. depressive speech, we load the best model checkpoint from training and do just testing:
	$ bash
	TOKENIZERS_PARALLELISM=false python -m src.train experiment=emnlp/finetuning/duration_finetuning train=False test=True ckpt_path="/logs/train/runs/FIND/MODEL/CHECKPOINT.ckpt" data.test_file=test_depression.pkl


9. Run compute_plot_MI.ipynb to:
	i. Estimate unconditional entropy (NLL) matching the distributional family for every feature used during regression.
	ii. Estimate conditional entropy based on the losses logged by the models during training and testing for each feature.
	iii. Compute mutual information per feature as MI = unconditional entropy - conditional entropy.
	iiii. Visualize time-scale of mutual information in heatmaps

10. Done! You can use data_summary.ipynb for statistics of the data used.

Changes to the original repo (https://github.com/Chief-Buka/contextual-redundancy):
 
-	New plotting notebook computing differential entropy using parametric distribution.
-	New notebook for estimating differential unconditional entropy using KDE. Use this for more robustness - not applied in the current repo setup. This can take some hours.
-	Added feature extraction notebook that utilizes modules in original repo for new data.
-	Added a New custom_pickle_datamodule, which allows for passing new data extracted using above feature extraction methods to the model regression of the original repo.
-	Added notebook for generating pkl files using dataframes from above steps. This notebook also generates new finetuning.yaml files for every feature in configs/experiment/emnlp/finetuning and new _datamodule.yaml files in configs/data/, that both use prior regression modules used in conjunction with custom pickle module
-	All TokenTaggingRegressor files have had import statements for AdamW change to torch.optim, memory_profiler import statements commented out in all these aswell.
- 	Settings have also been added that generates 4 context windows per utterance, to better utilize limited data.
-	torch_metrics.py & torch_metrics_cm.py have had MaskedPearsonCorrCoeff classes updated to not rely on local helper functions due to dependency conflicts.
-	updated requirements (split into audio_reqs.txt & training_reqs.txt for 2 envs used during full pipeline).
-	Added data summary notebook for producing statistics on the data

-

