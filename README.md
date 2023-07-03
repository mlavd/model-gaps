# Code Execution Capability as a Metric

The official reproduction repository for "Code Execution Capability as a Metric for Machine Learning--Assisted Software Vulnerability Detection Models".


## Installation

### Environments
Conda environments are provided in the `envs` directory. Because different models use different libraries, 5 separate environments are included. The table below lists each environments file, name, and the models that use it.

| File           | Name     | Models            |
|----------------|----------|-------------------|
| `codebert.yml` | codebert | CodeBERT, LineVul |
| `cotext.yml`   | cotext   | CoTeXT            |
| `regvd.yml`    | regvd    | ReGVD             |
| `textcnn.yml`  | textcnn  | TextCNN           |
| `xgboost.yml`  | xgboost  | XGBoost           |


#### Using Environments
These commands create all five environments.

```bash
conda env create -f envs/codebert.yml
conda env create -f envs/cotext.yml
conda env create -f envs/regvd.yml
conda env create -f envs/textcnn.yml
conda env create -f envs/xgboost.yml
```

To activate an environment, such as codebert:
```bash
conda activate codebert
```

### Singularity
We trained all our models on NVIDIA A100's available through Wright State University's computing resources. Slurm and Singularity are used to manage those resources. The Singularity definitions are provided in `/singularity`.

#### Building Singularity Images
Run these commands with a working [Singularity installation](https://docs.sylabs.io/guides/3.0/user-guide/installation.html) from the root of the repository to build the Singularity images. `sudo` is required.

```bash
cd singularity
sudo singularity build cotext.sif cotext.def
sudo singularity build lightning.sif lightning.def
sudo singularity build linevul.sif linevul.def
sudo singularity build regvd.sif regvd.def
sudo singularity build xgboost.sif xgboost.def
cd ..
```

## Datasets
Our paper makes use of several datasets. Due to size restrictions, we do not provide them in this repository. However, scripts are available to perform the necessary preprocessing. Further, we maintain a copy of the datasets that we would be happy to share directly with you. Simply email the corresponding author.

### Dataset Preparation
#### Code Execution Tasks
1. Run the following commands:

```bash
python jsonl_task_export.py -t 1
python jsonl_task_export.py -t 2
python jsonl_task_export.py -t 3
python jsonl_task_export.py -t 4
python jsonl_task_export.py -t 5
python jsonl_task_export.py -t 6
```

2. Verify the following three files are in each of the task directories `data/jsonl/task[1-6]`:
    * `test.jsonl`
    * `train.jsonl`
    * `valid.jsonl`

#### CodeXGLUE / Devign
1. Download the dataset using the instructions and scripts in the [CodeXGLUE repository](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection). You should have 3 files:
    * `test.jsonl`
    * `train.jsonl`
    * `valid.jsonl`
2. Move these files into `data/jsonl/codexglue`

#### D2A
1. Download the D2A Leaderboard Dataset (V1.0) from the [IBM Data Exchange](https://developer.ibm.com/exchanges/data/all/d2a/).
2. Place these three files into `data/input/d2a`:
    * `d2a_lbv1_function_dev.csv`
    * `d2a_lbv1_function_test.csv`
    * `d2a_lbv1_function_train.csv`
3. Run `python jsonl_d2a_export.py`. The following files should now be in `data/jsonl/d2a`:
    * `dev.jsonl`
    * `test.jsonl`
    * `train.jsonl`
4. Rename `dev.jsonl` to `valid.jsonl`.

#### Draper VDISC
1. Download the Draper VDISC dataset from [OSF.io](https://osf.io/d45bw/)
2. Place the three files into `data/input/draper`:
    * `VDISC_test.hdf5`
    * `VDISC_train.hdf5`
    * `VDISC_validate.hdf5`
3. Run `python jsonl_draper_export.py`. The following files should now be in `data/jsonl/draper`:
    * `test.jsonl`
    * `train.jsonl`
    * `validate.jsonl`
4. Rename `validate.jsonl` to `valid.jsonl`.

#### CodeXGLUE+D2A
1. Follow the steps to download and process CodeXGLUE and D2A.
2. Run the following commands:

```bash
python jsonl_merger.py --a=d2a/valid.jsonl --b=codexglue/test.jsonl --out=codexglued2a/test.jsonl
cp ./data/jsonl/codexglue/valid.jsonl ./data/jsonl/codexglued2a/valid.jsonl
python jsonl_merger.py --a=d2a/train.jsonl --b=codexglue/train.jsonl --out=codexglued2a/train.jsonl
```

3. Verify the following three files are in `data/jsonl/codexglued2a`:
    * `test.jsonl`
    * `train.jsonl`
    * `valid.jsonl`

#### All
1. Follow the steps to build CodeXGLUE+D2A
2. Run the following commands:

```bash
python jsonl_merger.py --a=draper/test.jsonl --b=codexglued2a/test.jsonl --out=all/test.jsonl
python jsonl_merger.py --a=draper/valid.jsonl --b=codexglued2a/valid.jsonl --out=all/valid.jsonl
python jsonl_merger.py --a=draper/train.jsonl --b=codexglued2a/train.jsonl --out=all/train.jsonl
```

3. Verify the following three files are in `data/jsonl/all`:
    * `test.jsonl`
    * `train.jsonl`
    * `valid.jsonl`


#### All Balanced
> **Note:** Because resampling requires the training split of the All dataset to first be converted to CodeBERT embeddings, which takes a significant amount of time and space, we include the indices of the samples used to accelerate the process.

1. Follow the steps to build All.
2. Run the following commands:

```bash
cp data/jsonl/all/test.jsonl data/jsonl/allbalanced/test.jsonl
cp data/jsonl/all/train.jsonl data/jsonl/allbalanced/train.jsonl
cp data/jsonl/all/valid.jsonl data/jsonl/allbalanced/valid.jsonl
python jsonl_sampler.py --jsonl=data/jsonl/allbalanced/train.jsonl --indices=data/jsonl/allbalanced/indices.txt
```

3. Verify the following three files are in `data/jsonl/allbalanced`:
    * `test.jsonl`
    * `train.jsonl`
    * `valid.jsonl`
4. Verify `train.jsonl` contains 156,818 using `cat data/jsonl/allbalanced/train.jsonl | wc -l`.


### XGBoost Dataset Preparation
XGBoost uses embeddings from CodeBERT instead of the JSONL files. To improve the speed of training, these embeddings are precalculated and cached. However, this can take a significant amount of space. We recommend only generating these datasets as they are needed. The following commands work for all datasets, simply replace the dataset variable as needed.

```bash
export $DS=codexglue
python embeddings_extractor.py --jsonl=data/jsonl/$DS/test.jsonl --output=data/embeddings/test
python embeddings_extractor.py --jsonl=data/jsonl/$DS/train.jsonl --output=data/embeddings/train
python embeddings_extractor.py --jsonl=data/jsonl/$DS/valid.jsonl --output=data/embeddings/valid
```


## Lessons Learned

What did you learn while building this project? What challenges did you face and how did you overcome them?


## Running Tests

To run tests, run the following command

```bash
  npm run test
```


## Contributing / Updates

As a reproduction repository, no updates will be made to this repository unless they are required to fix a bug in the code. Updates will only be made by the original authors. A log of updates will be provided here.

### Updates
- No updates have been made


## Features

- Light/dark mode toggle
- Live previews
- Fullscreen mode
- Cross platform


## Authors

- [@octokatherine](https://www.github.com/octokatherine)


## Acknowledgements
 - [Awesome Readme Templates](https://awesomeopensource.com/project/elangosundar/awesome-README-templates)
 - [Awesome README](https://github.com/matiassingers/awesome-readme)
 - [How to write a Good readme](https://bulldogjob.com/news/449-how-to-write-a-good-readme-for-your-github-project)

