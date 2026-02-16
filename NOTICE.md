# Open Source Notices

This project, **Qari Finder**, relies on open-source research, datasets, and libraries.

## Manhuw (Core Recognition Logic)

* **Original Project:** [manhuw](https://github.com/musa11971/manhuw)
* **Author:** Musa11971
* **License:** GPL-3.0
* **Usage:**
    * Reciter classification architecture.
    * Audio feature extraction parameters.
    * Initial training dataset structure.

## Datasets

* **ESC-50 (Environmental Sound Classification):**
    * **Source:** [ESC-50 on GitHub](https://github.com/karoldvl/ESC-50)
    * **Author:** Karol J. Piczak
    * **License:** [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
    * **Usage:** Used to train the `_background` class for noise and silence rejection.

## Libraries

* **Meyda:** MIT License (Audio Feature Extraction)
* **TensorFlow.js:** Apache 2.0 (Inference Engine)
* **Librosa:** ISC License (Python Audio Processing)
* **SoundFile:** BSD 3-Clause (Audio I/O)