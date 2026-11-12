# Sample input

The app reads the `eeg` port of **eeg-prep** (a BIDS-EEG derivative), any BIDS-EEG dataset, or
loose EDF files. Three ways to get real input:

1. **BIDS Siena Scalp EEG Database** (SzCORE's BIDS conversion; 14 adults with focal epilepsy,
   41 recordings, 141 h, 47 seizures as HED-SCORE events): Zenodo
   [10.5281/zenodo.10640762](https://doi.org/10.5281/zenodo.10640762), CC BY 4.0. Link one subject
   folder (`sub-00/ses-01/eeg/`: five recordings of 35-45 min, one seizure each) or the whole
   dataset to the `eeg` port. Original: PhysioNet
   [10.13026/5d4a-j060](https://doi.org/10.13026/5d4a-j060).
2. **CHB-MIT Scalp EEG Database** as PhysioNet ships it (bipolar-only EDFs plus
   `chbNN-summary.txt` with the seizure times): PhysioNet
   [10.13026/C2K01R](https://doi.org/10.13026/C2K01R), ODC-By 1.0, e.g.
   `https://physionet.org/files/chbmit/1.0.0/chb01/` (`chb01_03.edf`, `chb01_04.edf` and
   `chb01-summary.txt` hold two seizures).
3. **Your own EEG** through eeg-prep (any hospital's EDF, BrainVision, Nihon Kohden or Natus
   export), with seizure times in `labels.csv` if you want scores.

`make fixture` writes a small synthetic input (`tests/make_fixture.py`): two referential
recordings (one with two planted seizures, a short artifact and a marked bad segment; one
at 512 Hz without seizures) and one CHB-MIT-style bipolar-only recording with a planted seizure.
