

# Extending DomainBed to Audio Domain Generalization and Adaptation

An extension of [DomainBed](https://github.com/facebookresearch/DomainBed) for systematic evaluation of audio domain generalization and domain adaptation.

**Resources:**
[🤗 Hugging Face Checkpoints](HUGGINGFACE_URL) · [📄 DG in audio](ARXIV_PAPER_1_URL) · [📄 From DG to SDA/UDA](ARXIV_PAPER_2_URL) 


This repository extends the original [DomainBed](https://github.com/facebookresearch/DomainBed) framework to support audio domain and provides a unified framework for studying domain shifts from Domain Generalization (DG) to Supervised Domain Adaptation (SDA) and Unsupervised Domain Adaptation (UDA).


> **This repository is an extension of DomainBed.**
> The original DomainBed framework provides the foundation for the experimental pipeline, algorithms, and domain generalization setting. This repository builds upon that codebase by introducing audio-specific components and domain adaptation settings. Where possible, the implementation preserves the structure and conventions of DomainBed in order to make comparisons with existing DomainBed experiments easier.

---

## Overview

[DomainBed](https://github.com/facebookresearch/DomainBed)  is a framework for studying **Domain Generalization (DG)**, where a model is trained on one or more source domains and evaluated on an unseen target domain.

This project extends this setting to audio (CWWS dataset) and considers a broader continuum of domain shift scenarios:
<!-- 
```text
                                                        Domain Shift 
                                                    (VLCS, CWWS, PACS, ...)
                                                            │        
                                                            ▼        
                                                          Domain      
                                                      Generalization   
                                                           (DG)
                                                            │             
                                                        No target data 
                                                        during training
                                                            │
                                           ┌────────────────┼────────────────┐
                                           │                                 │
                                           ▼                                 ▼
                                       Supervised                       Unsupervised
                                       Adaptation                        Adaptation
                                         (SDA)                             (UDA)
                                           │                                 │
                                    Labeled target                   Unlabeled target
                                    data available                    data available
``` -->

```text
                              DomainBed
                                 │
                   ┌─────────────┴────────────────────┐
                   │                                  │
            Original framework                    Extensions
                   │                                  │
                   ▼                        ┌─────────┴─────────┐
            Domain Generalization           │                   │
            Experimental framework          ▼                   ▼
            Algorithms                 Audio Domain    DG → SDA & DG → UDA
            Evaluation                Generalization    Domain Adaptation
                                        Benchmark           Continuum
                                            │                   │
                                            ▼                   ▼
                                       📄 Paper 1            📄 Paper 2
```

📄 **Paper 1:** [Audio Domain Generalization and Dataset Validation](PAPER_1_URL)
📄 **Paper 2:** [From Domain Generalization to Supervised and Unsupervised Domain Adaptation](PAPER_2_URL)


---

## What is new?

The main extensions introduced in this repository are:

### 🎵 Audio modality

The original DomainBed framework is extended to operate on **audio data**, including the corresponding:

* Audio datasets and domain definitions
* Audio data loading and preprocessing
* Feature extraction / representations
* Audio-specific training pipelines
* Evaluation procedures

The framework considers six acoustic domains, ranging from the original clean speech data to different types of environmental noise, signal distortion, and speech coding artifacts.

### 🎧 Audio domains and examples

The following table summarizes the six domains considered in our experiments and provides representative audio examples.

> **Note:** The audio files are provided in `examples/audio/`. Click on an example to listen to the corresponding sample.

|      Domain      | Description                                                                                                                                                                                                  |                              Example 1                             |                              Example 2                             |
| :--------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------: | :----------------------------------------------------------------: |
|    **`clean`**    | The original **clean speech data**, without additional noise or signal distortion.                                                                                                                           |    [▶ Listen](examples/audio/clean/common_voice_de_19042713.mp3)    |    [▶ Listen](examples/audio/clean/common_voice_fr_17426200.mp3)    |
|    **`wham`**    | Additive noise with real-world background noise recorded in everyday environments from the **WHAM! dataset**. The noise consists primarily of conversational speech (*babble noise*). |    [▶ Listen](examples/audio/wham/common_voice_es_18306598.mp3)    |    [▶ Listen](examples/audio/wham/common_voice_zh-CN_18547567.mp3)    |
|    **`wind`**    | Additive noise generated using the **wind noise simulator**.                                                                                             |    [▶ Listen](examples/audio/wind/common_voice_de_19015428.mp3)    |    [▶ Listen](examples/audio/wind/common_voice_zh-CN_23866163.mp3)    |
| **`saturation`** | A nonlinear amplitude transformation inducing **signal clipping and distortion**, with the distortion gain sampled uniformly from $[0.5, 4.5]$.                                                              | [▶ Listen](examples/audio/saturation/common_voice_es_18311437.mp3) | [▶ Listen](examples/audio/saturation/common_voice_fr_17519345.mp3) |
|   **`demand`**   | Additive noise from the **DEMAND dataset** , providing real-world environmental noise recordings.                                                                                   |   [▶ Listen](examples/audio/demand/common_voice_fr_17300081.mp3)   |   [▶ Listen](examples/audio/demand/common_voice_fr_17300082.mp3)   |
|    **`lpc10`**   | Application of the **LPC10 speech codec** to the clean audio files, introducing speech coding artifacts.                                                                             |    [▶ Listen](examples/audio/lpc10/common_voice_fr_17300081.mp3)   |    [▶ Listen](examples/audio/lpc10/common_voice_fr_17300082.mp3)   |

For additive noises (`wham`, `wind`, and `demand`) we consider a range of possible SNRS (uniform distribution over [-10, 10] dB).


This diversity makes the benchmark suitable for studying **domain generalization and domain adaptation in audio**, including the transition from settings where the target domain is completely unseen to settings where target-domain data are available during training. These examples illustrate the diversity of acoustic conditions and domain shifts considered in this framework. The same underlying task can therefore be evaluated across substantially different acoustic domains, providing a testbed for **domain generalization and domain adaptation in audio**.



---


### 🌍 Domain Adaptation

In addition to the original Domain Generalization setting, this repository introduces domain adaptation scenarios where information from the target domain can be available during training.

The framework supports:

* **Domain Generalization (DG)**
  Target-domain data are not available during training.

* **Supervised Domain Adaptation (SDA)**
  Labeled samples from the target domain are available during training.

* **Unsupervised Domain Adaptation (UDA)**
  Unlabeled samples from the target domain are available during training.

This makes it possible to study how performance evolves as increasing amounts of target-domain information become available.

---


## Installation

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

Create the environment and install all dependencies specified in pyproject.toml:

```bash
uv sync
```

Depending on the experiment, additional dependencies may be required.

---

## Usage 

### Audio Dataset  

**Build your own audio dataset**  
Create your own dataset in .json format before building: 

```bash
uv run --no-sync python -m domainbed.prepare_data.create_json \
    --dataset_name noisy_CWWS \
    --path_to_speech_data $SCRATCH/domainbed/data/URGENT/corpus/CommonVoice/cv-corpus-22.0-2025-06-20  \
    --path_to_wham_noise $SCRATCH/domainbed/data/URGENT/noises/wham_noise \ 
    --output_path domainbed/prepare_data/save_json
```

Build the audio dataset from .json file: 

```bash
uv run python -m domainbed.prepare_data.rebuild_dataset \ 
    --dataset_info domainbed/prepare_data/save_json/noisy_CWWS_train_small.json \
    --path_to_wham_noise $SCRATCH/domainbed/data/URGENT/noises/wham_noise \ 
    --commonvoice_root $SCRATCH/domainbed/data/URGENT/corpus/CommonVoice/cv-corpus-22.0-2025-06-20 \
    --output_root domainbed/prepare_data/datasets \
    --debug
```

To continue and perform DG/DA experiments on the CWWSaudio dataset, you should generate train_small (noisy_CWWS_train_small), train_mono (noisy_CWWS_train_mono), validation (noisy_CWWS_validation) and test (noisy_CWWS_test) splits.

```bash
uv run python -m domainbed.prepare_data.rebuild_dataset \ 
    --dataset_info domainbed/prepare_data/noisy_CWWS_train_small.json \
    --path_to_wham_noise $SCRATCH/domainbed/data/URGENT/noises/wham_noise \ 
    --commonvoice_root $SCRATCH/domainbed/data/URGENT/corpus/CommonVoice/cv-corpus-22.0-2025-06-20 \
    --output_root domainbed/prepare_data/datasets \
    --debug
```


**Rerproduce the dataset used in the paper**  
To build the splits used in the paper, please generate from the json files in: ``reproduce_dataset`` folder 

```bash
uv run python -m domainbed.prepare_data.rebuild_dataset \ 
    --dataset_info domainbed/prepare_data/reproduce_dataset/noisy_CWWS_train_small.json \
    --path_to_wham_noise $SCRATCH/domainbed/data/URGENT/noises/wham_noise \ 
    --commonvoice_root $SCRATCH/domainbed/data/URGENT/corpus/CommonVoice/cv-corpus-22.0-2025-06-20 \
    --output_root domainbed/prepare_data/datasets \
    --debug
```


### Experiments
The general workflow follows the organization of DomainBed for DG experiments:

Domain Generalization:

```bash
uv run python -m domain.scripts.train \
    --algorithm <ALGORITHM> \
    --dataset <DATASET> \
    --test_env <TARGET_DOMAIN>
```

> SDA and UDA experiments need DG experiments to build the continuum by using general hyperparameters selected from Domain Generalization experiments. This allow a continuous study. 

Supervised Domain Adaptation: 


```bash
uv run python -m domain.scripts.train \
    --algorithm <ALGORITHM> \
    --dataset <DATASET> \
    --test_env <TARGET_DOMAIN>
```

Unsupervised Domain Adaptation : 
 

```bash
uv run python -m domain.scripts.train \
    --algorithm <ALGORITHM> \
    --dataset <DATASET> \
    --test_env <TARGET_DOMAIN>
```
--- 

Generate sweeps over hyperparemeters

```bash
uv run python -m domain.scripts.train \
    --algorithm <ALGORITHM> \
    --dataset <DATASET> \
    --test_env <TARGET_DOMAIN>
```

Extract embeddings: 

You can extract embeddings from experiments already computed and you should chose a model selection approach:

```bash
uv run python -m domain.scripts.train \
    --algorithm <ALGORITHM> \
    --dataset <DATASET> \
    --test_env <TARGET_DOMAIN>
```

See the `scripts/` directory for experiment configurations and examples.

> **Note:** The exact commands and available arguments may differ from the original DomainBed implementation because of the additional audio and domain adaptation functionality.

---

## Reproducibility Statement

Experiments are organized following the experimental philosophy of DomainBed whenever possible.

1. Build audio dataset 
2. Launch DG experiments 
3. Launch DA experiments from DG baseline 
4. extract features 

This allows results obtained with this repository to be compared with the corresponding DomainBed-style experiments.


## 📚 Publications using this repository

This repository contains the code and experimental framework developed and used in the following publications. The work progressively extends the original [DomainBed](https://github.com/facebookresearch/DomainBed) framework from image-based Domain Generalization to audio-domain research and, subsequently, to a broader continuum encompassing Domain Generalization, Supervised Domain Adaptation, and Unsupervised Domain Adaptation.

### 1. Audio Domain Generalization and Dataset Validation

**[Paper Title]**
*Authors*
*Venue, Year*

[📄 Paper](PAPER_URL) · [🔗 DOI](DOI_URL)

This work introduces and validates the proposed **audio domain benchmark** through a systematic study of **Domain Generalization (DG)**. The experiments investigate whether models trained on multiple audio domains can generalize to previously unseen domains under controlled domain shifts.

The experiments use the audio domains provided in this repository. This publication provides the initial validation of the audio benchmark and establishes the foundation for the subsequent domain adaptation experiments.

### 2. From Domain Generalization to Domain Adaptation

**[Paper Title]**
*Authors*
*Venue, Year*

[📄 Paper](PAPER_URL) · [🔗 DOI](DOI_URL)

Building upon the first study, this work extends the framework beyond **Domain Generalization** to investigate a broader **continuum of domain shift scenarios**, ranging from DG to:

* **Domain Generalization (DG)** — no target-domain data are available during training.
* **Supervised Domain Adaptation (SDA)** — labeled target-domain data are available during adaptation.
* **Unsupervised Domain Adaptation (UDA)** — unlabeled target-domain data are available during adaptation.

The repository provides a unified experimental framework for studying these settings using the same audio domains, data-processing pipeline, and evaluation methodology. This enables a more systematic comparison of how different levels of access to target-domain information affect model adaptation and generalization.

---

## Citation

If you use the original DomainBed framework, please cite the corresponding DomainBed publication:

```bibtex
@inproceedings{gulrajani2021in,
  title={In Search of Lost Domain Generalization},
  author={Gulrajani, Ishaan and Lopez-Paz, David},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```

If you use this extension in your research, please also cite this repository / accompanying publication:

```bibtex
@misc{<YOUR_CITATION_KEY>,
  title={DomainBed-Audio: ...},
  author={<YOUR_NAME>},
  year={<YEAR>},
  url={<YOUR_REPOSITORY_URL>}
}
```

```bibtex
@misc{<YOUR_CITATION_KEY>,
  title={DomainBed-Audio: ...},
  author={<YOUR_NAME>},
  year={<YEAR>},
  url={<YOUR_REPOSITORY_URL>}
}
```

---

## Acknowledgements

This project builds upon the excellent work of the authors of **DomainBed**.

We thank the DomainBed authors for making their framework and implementation publicly available.

Original repository:

https://github.com/facebookresearch/DomainBed

This repository should therefore be understood as an **extension of the DomainBed framework**, with additional functionality for audio and domain adaptation.

---

## License

This project follows the licensing terms of the original DomainBed code where applicable.

Additional code introduced by this repository is released under:

**[INSERT YOUR LICENSE HERE]**

Please consult the original DomainBed repository and the individual files for applicable licensing and attribution requirements.

---

## Project Status

This repository is under active development.

Current extensions include:

* [x] Audio modality
* [x] Domain Generalization
* [x] Supervised Domain Adaptation
* [x] Unsupervised Domain Adaptation

---

## Contact

For questions, issues, or contributions, please open an issue or pull request on this repository.

If you use this code in a publication, please consider citing both the original **DomainBed** work and this extension.
