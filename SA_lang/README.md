Directory that contains all of the ShapeAssembly specific logic that was used for the ShapeMOD experiments.

None of the logic in this directory is need to run the novel shape generation / visual program induction experiments (see the task/ folder for that).

# Data: Parsing PartNet Logic .prog files

ShapeMOD.py consumes a .prog file that defines a dataset of programs.

All of the logic for parsing PartNet into a dataset of program can be found in the parse_files directory.

Pre-comptuted .prog files can be found at:

```
chair.prog : https://drive.google.com/file/d/1mjXcnjh9hlT4eXqQ1su1x_h643IdafwP/view?usp=sharing
table.prog : https://drive.google.com/file/d/1nAVTCMu4xV9luXhgjVYGOTryl2Zwo0Z9/view?usp=sharing
storage.prog : https://drive.google.com/file/d/1Qc6GmmSQrfSfEJN73yxHo4obc3Xs3fjT/view?usp=sharing
```

To regenerate .prog files you can run a command like:

```
python3 make_prog_dataset.py parsed_data/chair_data/ program_datasets/chair.prog 100000
```

This will require pickled data parsed from PartNet into the parsed_data folder.

These can be found as .zip files in that folder or they can be re-generated with a command like (run from parse_files directory):

```
python3 json_parse.py chair ../parsed_data/chair ../tasks/data/chair/
```

# Data: Training models on .prog files

The models in the tasks folder consume .data and .meta files that are created from a .prog file and a DSL discovered by ShapeMOD.

The .meta file describes properties of the discovered library and the .data file describes the best program for each shape instance in the .prog dataset under the discovered DSL.

See the README in the tasks folder to download pre-generated .data and .meta files, or they can be re-generated with a command like (run from tasks directory):

```
python3 make_abs_data.py data/shapemod_chair ../program_datasets/chair.prog chair shapemod_chair_dsl sa_config
```

# Explanation of files

**make_prog_dataset.py** -> logic for turning parsed data into .prog files (main entrypoint within this directory)

**parsed_data/** --> parsed data that can be converted into .prog files

**parse_files/** --> logic to turn PartNet into pickled parsed data

**program_datasets/** --> where .prog files live

**sa_config.py** --> defines how ShapeMOD should operate over the ShapeAssembly language. Details described in the comments of this file

**sa_prog_order.py** --> find all valid ordering of program attributes given a PartNet parse

**sa_utils.py** --> util functions that are ShapeAssembly specific

**tasks/** --> logic for modeling experiments