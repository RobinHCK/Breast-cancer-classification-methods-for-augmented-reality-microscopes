<!-- TITLE -->
<br />
<p align="center">
  <h3 align="center">Breast cancer classification methods for augmented reality microscopes</h3>
</p>

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Presentation of the Project](#presentation-of-the-project)
* [Prerequisite](#prerequisite)
* [Workflow](#workflow)
  * [Train](#train)
  * [Test](#test)
  * [Evaluate](#evaluate)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)
* [References](#references)


<!-- PRESENTATION OF THE PROJECT -->
## Presentation Of The Project

This work is a deep learning project applied to medical images. 
We propose a real-time method robust to changes in magnification levels by training deep neural networks Inception-ResNet-V2 [1] on certain key levels of Whole Slide Images (WSI) and testing them on the levels of a microscope. 


<!-- GETTING STARTED -->
## Prerequisite

- Before executing the sctipts, make sure you have correctly edited the configuration file: **config.cfg**
- The medical images used in this project come from [BreakHis](https://www.kaggle.com/datasets/ambarish/breakhis) dataset proposed in [2]
![breakhis](https://github.com/RobinHCK/Breast-cancer-classification-methods-for-augmented-reality-microscopes/blob/main/img/breakhis.png)


<!-- WORKFLOW -->
## Workflow

### Train

- Train one network per magnification level and per fold:
  - python 1_experiment_train.py
- Train one network on all magnification levels and per fold:
  - python 2_experiment_train_all.py

### Test

- Test every network on every magnification level per fold:
  - python 3_experiment_test.py
- Test the networks trained on all magnification levels on every magnification level per fold:
  - python 4_experiment_test_all.py

### Evaluate

- Create the heatmap:
  - python 5_experiment_heatmap.py 
![heatmap](https://github.com/RobinHCK/Breast-cancer-classification-methods-for-augmented-reality-microscopes/blob/main/img/heatmap.png)
- Create accuracy and loss graph:
  - python 6_save_graphs.py 
- Compare methods thanks to the results shown in the heatmap:
![methods](https://github.com/RobinHCK/Breast-cancer-classification-methods-for-augmented-reality-microscopes/blob/main/img/methods.png)

<!-- CONTACT -->
## Contact

Robin Heckenauer - robin.heckenauer@gmail.com

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

The High Performance Computing center of the University of Strasbourg. The computing resources were funded by the Equipex Equip@Meso project (Programme Investissements d'Avenir) and the CPER Al

<!-- REFERENCES -->
## References

[1] Szegedy, Christian, et al. "Inception-v4, inception-resnet and the impact of residual connections on learning." Thirty-first AAAI conference on artificial intelligence. 2017.

[2] Spanhol, Fabio A., et al. "A dataset for breast cancer histopathological image classification." Ieee transactions on biomedical engineering 63.7 (2015): 1455-1462.
