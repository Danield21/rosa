# ROSA: Random Orthogonal Subspace Adaptation
This repository is the official implementation of [ROSA: Random Orthogonal Subspace Adaptation](https://openreview.net/forum?id=4P9vOFpb63). 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
pip install -U datasets // update datasets library
pip install git+https://github.com/huggingface/transformers //
```

## Training

To train ROSA/LoRA model(s) on GLUE benchmark run this command:

```commandline
python train_mlm.py 
    dataset.cache=/path/to/save/huggingface/dataset
    output.path=/path/to/save/checkpoints
    +task=cola # sst2, mrpc, stsb, qqp, mnli, qnli, rte, wnli
    model.name=gpt2  # gpt2, gpt2-medium, gpt2-large, gpt2-xl
    train.batch_size=16 
    train.lr=2e-5
    fnmodel.name=rosa # lora
    fnmodel.params.rank=8 
    fnmodel.params.factorize_method=svd_equal
```

## Evaluation
### Visualize train/validation curves of model(s)
Run the following command to visualize the train/validation curves of model(s) in the paper:

```commandline
tensorboard --logdir=/path/to/saved/runs
```

[//]: # (### Plot figures of model&#40;s&#41; in the paper)

[//]: # (To create figures of model&#40;s&#41; in the paper for e2e or eli5 dataset, run:)

[//]: # ()
[//]: # (```commandline)

[//]: # (python figures.py --fn /path/to/saved/runs/for/dataset/<dataset_name>)

[//]: # (```)

[//]: # (with `<dataset_name>` being `e2e_nlg` or `eli5`.)


## Citation
If you find this repository useful in your research, please cite our paper:

```bibtex
@inproceedings{hameed2023rosa,
  title={ROSA: Random Orthogonal Subspace Adaptation},
  author={Marawan Gamal Abdel Hameed and Guillaume Rabusseau}
  maintitle = {International Conference on Machine Learning},
  booktitle = {Efficient Systems for Foundation Models},
  year={2023}
}
```
