# Music Data Processing Project

## Overview
This project involves processing and analyzing a music dataset, including MIDI files, lyrics, and their metadata.

## Datasets
The project utilizes the following datasets:
- **LMD Aligned MIDI Files**: Contains MIDI files aligned with corresponding audio files. 
  - Download: [lmd_aligned.tar.gz](http://hog.ee.columbia.edu/craffel/lmd/lmd_aligned.tar.gz)
- **Match Scores File**: Provides matching scores between MIDI files and other dataset elements.
  - Download: [match_scores.json](http://hog.ee.columbia.edu/craffel/lmd/match_scores.json)
- **LMD Matched H5 Files**: Consists of Hierarchical Data Format version 5 (H5) files matched to the MIDI data.
  - Download: [lmd_matched_h5.tar.gz](http://hog.ee.columbia.edu/craffel/lmd/lmd_matched_h5.tar.gz)

## Installation
Before running the project, ensure you have the following dependencies installed:

### Python Libraries
Use `pip3` to install the required Python libraries:

pip3 install librosa
pip3 install mir_eval
pip3 install IPython
pip3 install tables
pip3 install numpy
pip3 install matplotlib
pip3 install pandas
pip3 install pretty_midi
pip3 install tables
pip3 install scikit-learn
pip3 install nltk

## Example Usage
Example usage: python3 ./database.py