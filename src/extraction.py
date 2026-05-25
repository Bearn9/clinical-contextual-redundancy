import os
import nltk
import tgt # pip install tgt
from collections import OrderedDict
from tqdm import tqdm
from nltk.corpus import cmudict

# --- STEP 1: FIX REPOSITORY LOGIC (MONKEYPATCHING) ---
import src.utils.text_processing as tp
from src.data.components.feature_extractors import ProsodyFeatureExtractor

# Dummy CELEX reader
class DummyCelex:
    def __init__(self, *args, **kwargs):
        self.data = {}

    def get_syllable_count(self, word):
        return 1

tp.CelexReader = DummyCelex


def process_daic_files(self, lab_root=None, wav_root=None, verbose=False):

    if lab_root:
        self.lab_root = lab_root
    if wav_root:
        self.wav_root = wav_root

    self.samples = []

    for reader in tqdm(os.listdir(self.lab_root), desc="Processing DAIC-WOZ"):

        reader_path = os.path.join(self.lab_root, reader)

        if not os.path.isdir(reader_path):
            continue

        for ut_file in os.listdir(reader_path):

            if not ut_file.endswith(".TextGrid"):
                continue

            ut_id = ut_file.replace(".TextGrid", "")

            ut_path_lab = os.path.join(reader_path, ut_file)
            ut_path_wav = os.path.join(self.wav_root, reader, ut_id + ".wav")

            if not os.path.exists(ut_path_wav):
                continue

            tg = tgt.io.read_textgrid(ut_path_lab)
            words_tier = tg.get_tier_by_name("words")

            text = " ".join(
                i.text for i in words_tier if i.text not in ["", "sil", "sp"]
            )

            import tempfile

            tg = tgt.io.read_textgrid(ut_path_lab)
            words_tier = tg.get_tier_by_name("words")

            tmp_lab = tempfile.NamedTemporaryFile(delete=False, suffix=".lab")

            with open(tmp_lab.name, "w", encoding="utf8") as f:
                for w in words_tier:
                    if w.text in ["", "sil", "sp"]:
                        continue
                    f.write(f"{w.start_time} {w.end_time} {w.text}\n")

            features, _ = self._extract_features(
                lab_path=tmp_lab.name,
                wav_path=ut_path_wav,
                phoneme_lab_path=tmp_lab.name
            )

            if features is None:
                continue

            self.samples.append(OrderedDict({
                "reader": reader,
                "book": "na",
                "text": text,
                "features": features,
                "path_lab": ut_path_lab,
                "path_wav": ut_path_wav,
                "filename": ut_id
            }))

    return self.samples


# PATCH CLASS METHOD
ProsodyFeatureExtractor.process_files = process_daic_files

# --- STEP 2: SYLLABLE COUNTING ---
try:
    d = cmudict.dict()
except LookupError:
    nltk.download('cmudict')
    d = cmudict.dict()

def get_sylls(word):
    w = word.lower().strip(".,!?;:\"")
    return max(1, len([p for p in d[w] if p[-1].isdigit()])) if w in d else 1

# --- STEP 3: MAIN LOOP ---
def run_replication():
    categories = ['train-clean', 'test-clean', 'dev-clean']
    for cat in categories:
        LAB_ROOT = f"D:\\thesis_data\\DAIC-WOZ\\RootSplits\\aligned\\{cat}"
        WAV_ROOT = f"D:\\thesis_data\\DAIC-WOZ\\RootSplits\\wav\\{cat}"
        OUTPUT_FILE = f"D:\\thesis_data\\contextual-redundancy\\data\\{cat}_final.txt"
        
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        extractor = ProsodyFeatureExtractor(
            lab_root=LAB_ROOT,
            wav_root=WAV_ROOT,
            phoneme_lab_root=LAB_ROOT,
            celex_path="dummy",
            features=["f0","energy","duration","prominence"]
        )

        print(f"Extracting CWT features for {cat}...")
        results = extractor.extract_all() # Internally uses Suni et al. (2017) [3]

        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for utt_id, word_list in results.items():
                f.write(f"<file> {utt_id}\n")
                for w in word_list:
                    # Metric: Duration per Syllable (Regev et al. 2025 benchmark) [4]
                    dur = (w['end'] - w['start']) / get_sylls(w['word'])
                    prom = w.get('prominence', 0.0) # CWT LoMA Strength 
                    f.write(f"{w['word']}\t0\t0\t{dur:.4f}\t{prom:.4f}\n")
                f.write("\n")
    print("Pre-processing complete. Data ready for Redundancy Training.")

if __name__ == "__main__":
    run_replication()