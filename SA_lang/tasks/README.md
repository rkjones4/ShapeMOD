This directory contains ShapeAssembly specific experiment logic for the ShapeMOD paper.

# Data

DSLs for chair / table / storage categories can be found in the dsls/ folder. The cross-category (Section 6.5) DSL is also there.

Training on a category requires a .meta and .data file, these  can be downloaded, check README in data/ folder

See instructions in SA_lang repo to re-generate .meta and .data files for a specific category given a DSL and .prog file (using make_abs_data.py).

Any experiments on the visual program induction task require point cloud data, check the README in the pc_data folder.

Pre-trained models can be found at: https://drive.google.com/file/d/18GYzyJlHJvCVbBxi-VQGhRh68xKsq9i-/view?usp=sharing . Please put these folders in the model_output directory. gen_{} are the novel shape generation models (model_prog.py). infer_{} are the visual program models (infer_model_prog.py). base models use ShapeAssembly with no macros. shapemod models use the macros discovered by ShapeMOD. 

# Novel Shape Generation

model_prog.py is the entrypoint for the novel shape generation experiments, use the --help flag to see more details.

Sample new programs from a pre-trained generative model with a command like:

```
python3 model_prog.py -en model_output/gen_shapemod_chair/ -ng 10 -m gen -ds data/shapemod_chair -le 4499
```

Train a new model with a command like:
```
python3 model_prog.py -en my_experiment -ds data/shapemod_chair -c chair
```

# Visual Program Induction

The P.C. encoder we use can be found in pc_encoder.py . This encoder may require rebuilding. Try running python3 pc_encoder.py -> if do not see the output torch.Size([10, 256]), please see the README in the pointnet2 directory.

infer_model_prog.py is the entrypoint for the visual program induction experiments, use the --help flag to see more details.

Infer program from a pre-trained model with a command like:

```
python3 infer_model_prog.py -en model_output/infer_shapemod_chair/ -ng 10 -m infer -ds data/shapemod_chair -le 1449 -c chair
```

Train a new model with a command like:

```
python3 infer_model_prog.py -en my_experiment -ds data/shapemod_chair -c chair
```

# Explanation of files

**data/** -> where .data/.prog files live (output of make_abs_data.py) + where the .txt files of original ShapeAssembly programs can be found

**data_splits/** -> splits for novel shape generation between train / validation

**dsls/** -> discovered libraries from ShapeMOD

**etw_pytorch_utils.py** -> helper function for PointNet++ encoder

**gen_metrics.py** -> logic to calculate metrics for novel shape generation task

**infer_model_prog.py** -> visual program indufction model (e.g. point cloud -> program)

**infer_recon_metrics.py** -> reconstruction based metric calculation for point clouds

**infer_sem_valid.py** -> semantic validity program creation logic for infer_model_prog.py

**losses.py** -> logic where loss functions are defined

**make_abs_data.py** -> converts a .prog files into .data and .meta files using logic from ShapeMOD.py and a given DSL (e.g. by finding best program under DSL for each shape instance).

**model_output/** -> contains pre-trained models, also where new experiments are written to

**model_prog.py** -> novel shape generation model

**pc_data/** -> where point clouds from PartNet should be placed, see: pc_sample_partnet.py

**pc_data_splits.py** -> train/val/test splits for p.c. reconstruction tasks

**pc_encoder.py** -> PointNet++ encoder definition

**pc_sample_partnet.py** -> parse PartNet dataset and get point clods

**pointnet2/** -> more helper function for PointNet++ model, see README

**pointnet_fd.py** -> calculate frechet distance metric, using checkpoint of classification model in data/

**recon_metrics.py** -> logic to calculate metrics for validation set reconstruction of novel shape generation model

**sem_valid.py** -> semantic validity program creation logic for model_prog.py

**ShapeAssembly.py** -> defining ShapeAssembly language / executor

**valid.py** -> calculate physical validity metrics