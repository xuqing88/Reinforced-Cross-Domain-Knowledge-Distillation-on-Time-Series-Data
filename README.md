# Reinforced-Cross-Domain-Knowledge-Distillation-on-Time-Series-Data
This repository is a PyTorch implementation for NIPS 2024 Paper "Reinforced Cross-Domain Knowledge Distillation on Time Series Data".

## Datasets

### Available Datasets
We used four public datasets in this study. We also provide the **preprocessed** versions as follows:

- [UCIHAR](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/0SYHTZ)
- [HHAR](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/OWDFXO)
- [FD](https://mb.uni-paderborn.de/en/kat/main-research/datacenter/bearing-datacenter/data-sets-and-download)
- [SSC](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/UD1IM9)

Please download these datasets and put them in the respective folder in "data"


## Unsupervised Domain Adaptation Algorithms
### Existing Benchmark Algorithms
- [KD-STDA](https://arxiv.org/pdf/2101.07308)
- [KA-MCD](https://arxiv.org/pdf/1702.02052)
- [MLD-DA](https://openaccess.thecvf.com/content/WACV2021W/AVV/papers/Kothandaraman_Domain_Adaptive_Knowledge_Distillation_for_Driving_Scene_Semantic_Segmentation_WACVW_2021_paper.pdf)
- [REDA](https://junguangjiang.github.io/files/resource-efficient-domain-adaptation-acmmm20.pdf)
- [AAD](https://arxiv.org/pdf/2010.11478.pdf)
- [MobileDA](https://ieeexplore.ieee.org/abstract/document/9016215/)
- [UNI-KD](https://arxiv.org/pdf/2307.03347)

## Runing Proposed RCD-KD Algorithm

### Teacher training
Our approach requires a pre-trained teacher. We utilize DANN method to train a teacher and store them in 'experiments_logs/HAR/Teacher_CNN'.
For different dataset, please save the teachers into respectively dataset folder. Note that for teacher, we set 'feature_dim = 64' in 'configs/data_model_configs.py' 
and for the student we set 'feature_dim = 16'.

## Student training
To train a student with our proposed approach, run:

```
python proposed_RCD_KD.py  --experiment_description exp1  \
                --run_description run_1 \
                --da_method RL_JointADKD \
                --dataset HAR \
                --backbone CNN \
                --num_runs 3 \
```

## Claims
Part of benchmark methods code are from [AdaTime](https://github.com/emadeldeen24/AdaTime) and [UNI-KD](https://arxiv.org/pdf/2307.03347)
