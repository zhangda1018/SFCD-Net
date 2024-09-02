<div align="center">
    <h2>
        Integrating SAM with Feature Interaction for Remote Sensing Change Detection
    </h2>
</div>

[//]: # (<div align="center">)

[//]: # (  <img src="resources/RSPrompter.png" width="800"/>)

[//]: # (</div>)

<div align="center">
&nbsp;&nbsp;&nbsp;&nbsp;
</div>

<div align="center">

</div>

## Introduction

The repository is the code implementation of the paper [ Integrating SAM with Feature Interaction for Remote Sensing Change Detection](1), based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) and [TTP](https://github.com/KyanChen/TTP) projects.

The current branch has been tested under PyTorch 2.x and CUDA 12.1, supports Python 3.7+, and is compatible with most CUDA versions.

If you find this project helpful, please give us a star ⭐️, your support is our greatest motivation.


## Installation

### Dependencies

- Linux or Windows
- Python 3.7+, recommended 3.10
- PyTorch 2.0 or higher, recommended 2.1
- CUDA 11.7 or higher, recommended 12.1
- MMCV 2.0 or higher, recommended 2.1

### Environment Installation

We recommend using Miniconda for installation. The following command will create a virtual environment named `sfcd` and install PyTorch and MMCV.

Note: If you have experience with PyTorch and have already installed it, you can skip to the next section. Otherwise, you can follow these steps to prepare.

<details>

**Step 0**: Install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/index.html).

**Step 1**: Create a virtual environment named `sfcd` and activate it.

```shell
conda create -n sfcd python=3.10 -y
conda activate sfcd
```

**Step 2**: Install [PyTorch2.1.x](https://pytorch.org/get-started/locally/).

Linux/Windows:
```shell
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```
Or

```shell
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```

**Step 3**: Install [MMCV2.1.x](https://mmcv.readthedocs.io/en/latest/get_started/installation.html).

```shell
pip install -U openmim
mim install mmcv==2.1.0
```

**Step 4**: Install other dependencies.

```shell
pip install -U wandb einops importlib peft==0.8.2 scipy ftfy prettytable torchmetrics==1.3.1 transformers==4.38.1
```


</details>

### Install SFCD


Download or clone the TTP repository.

```shell
git clone git@github.com:zhangda1018/SFCD-Net.git
cd SFCD
```

## Dataset Preparation

<details>

### Levir-CD Change Detection Dataset

#### Dataset Download

- Image and label download address: [Levir-CD](https://chenhao.in/LEVIR/).

#### Organization Method

You can also choose other sources to download the data, but you need to organize the dataset in the following format:

```
${DATASET_ROOT} # Dataset root directory, for example: /home/username/data/levir-cd
├── train
│   ├── A
│   ├── B
│   └── label
├── val
│   ├── A
│   ├── B
│   └── label
└── test
    ├── A
    ├── B
    └── label
```

Note: In the project folder, we provide a folder named `data`, which contains an example of the organization method of the above dataset.

### Other Datasets

If you want to use other datasets, you can refer to [MMSegmentation documentation](https://mmsegmentation.readthedocs.io/zh-cn/latest/user_guides/2_dataset_prepare.html) to prepare the datasets.
</details>

## Model Training

### SFCD Model

#### Config File and Main Parameter Parsing

We provide the configuration files of the SFCD model used in the paper, which can be found in the `configs/SFCD` folder. The Config file is completely consistent with the API interface and usage of MMSegmentation. Below we provide an analysis of some of the main parameters. If you want to know more about the meaning of the parameters, you can refer to [MMSegmentation documentation](https://mmsegmentation.readthedocs.io/zh-cn/latest/user_guides/1_config.html).
<details>


#### Single Card Training

```shell
python tools/train.py configs/SFCD/xxx.py  # xxx.py is the configuration file you want to use
```


## Model Testing

#### Single Card Testing:

```shell
python tools/test.py configs/SFCD/xxx.py ${CHECKPOINT_FILE}  # xxx.py is the configuration file you want to use, CHECKPOINT_FILE is the checkpoint file you want to use
```


**Note**: If you need to get the visualization results, you can uncomment `default_hooks-visualization` in the Config file.


## Image Prediction

#### Single Image Prediction:

```shell
python demo/image_demo_with_cdinferencer.py ${IMAGE_FILE1} ${IMAGE_FILE2} configs/SFCD/image_infer.py --checkpoint ${CHECKPOINT_FILE} --out-dir ${OUTPUT_DIR}  # IMAGE_FILE is the image file you want to predict, xxx.py is the configuration file, CHECKPOINT_FILE is the checkpoint file you want to use, OUTPUT_DIR is the output path of the prediction result
```


## Acknowledgements

The repository is the code implementation of the paper [Integrating SAM with Feature Interaction for Remote Sensing Change Detection](1), based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) and [TTP](https://github.com/KyanChen/TTP) projects.

## Citation

If you use the code or performance benchmarks of this project in your research, please refer to the following bibtex to cite SFCD.

```
```

## License

The repository is licensed under the [Apache 2.0 license](LICENSE).

