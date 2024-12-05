# Automatic Music Transcription (AMT) Post-Processing Pipeline

This project is a post-processing pipeline for the CREPE model, designed to enhance pitch detection accuracy in automatic music transcription tasks. The project leverages the ChoralSingingDataset to evaluate and refine transcription results via rhythm quantization. We then generate a MusicXML file that contains the transcription of the given source audio.

## Features

- **Pitch Detection**: Utilizes the CREPE model for initial pitch detection.
- **Dataset**: Uses the ChoralSingingDataset for testing and validation.
- **Post-Processing**: Rhythm quantization to refine estimated MIDI piano roll results.

## Environment

- **Python Version**: 3.10
- **Main Notebook**: `AMT.ipynb`

## Requirements

All necessary packages and dependencies are listed in `requirements.txt`. Notable libraries include:
- [PrettyMIDI](https://github.com/craffel/pretty-midi)
- [mido](https://github.com/mido/mido)
- [crepe_notes](https://github.com/xavriley/crepe_notes)

## Usage

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open `AMT.ipynb` in a Jupyter Notebook-friendly environment to explore the project.

## Dataset

The project utilizes the [ChoralSingingDataset](https://example.com/choralsingingdataset) for evaluating transcription accuracy.
