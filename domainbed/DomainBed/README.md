

# Extending DomainBed to Audio Domain Generalization and Adaptation

An extension of [DomainBed](https://github.com/facebookresearch/DomainBed) for systematic evaluation of audio domain generalization and domain adaptation.

--- 


# 🗓️ Planned Release by early october 2026




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
   
```


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
|   **`demand`**   | Additive noise from the **DEMAND dataset** , providing real-world environmental noise recordings.                                                                                   |   [▶ Listen](examples/audio/demand/train-am_et-13-7103990432946619387.flac)   |   [▶ Listen](examples/audio/demand/train-am_et-28-15462378756025626659.flac)   |
|    **`lpc10`**   | Application of the **LPC10 speech codec** to the clean audio files, introducing speech coding artifacts.                                                                             |    [▶ Listen](examples/audio/lpc10/train-am_et-14-6306093779729425028.flac)   |    [▶ Listen](examples/audio/lpc10/train-am_et-28-15462378756025626659.flac)   |

For additive noises (`wham`, `wind`, and `demand`) we consider a range of possible SNRS (uniform distribution over [-10, 10] dB).


This diversity makes the benchmark suitable for studying **domain generalization and domain adaptation in audio**, including the transition from settings where the target domain is completely unseen to settings where target-domain data are available during training. These examples illustrate the diversity of acoustic conditions and domain shifts considered in this framework. The same underlying task can therefore be evaluated across substantially different acoustic domains, providing a testbed for **domain generalization and domain adaptation in audio**.



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

---

## Contact

For questions, issues, or contributions, please open an issue or pull request on this repository.

If you use this code in a publication, please consider citing both the original **DomainBed** work and this extension.
