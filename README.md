# ShapeMOD: Macro Operation Discovery for 3D Shape Programs

By [R. Kenny Jones](https://rkjones4.github.io/), [David Charatan](https://davidcharatan.com/), [Paul Guerrero](https://paulguerrero.net/), [Niloy J. Mitra](http://www0.cs.ucl.ac.uk/staff/n.mitra/), and [Daniel Ritchie](https://dritchie.github.io/)

![Overview](https://rkjones4.github.io/img/shapeMOD/teaser.png)

We propose ShapeMOD, an algorithm which takes as input a collection of 3D shape programs and makes them more compact by automatically discovering common macros which can be re-used across the collection.We apply ShapeMOD to datasets of ShapeAssembly programs and find that generative models which train on refactored programs containing these macros produce more plausible output shapes than those trained on the original programs. The discovered macros also facilitate shape editing by exposing only a small number of meaningful parameters for manipulating shape attributes. For example, the four_leg_base macro exposes two parameters (visualized as sliders with red handles); one parameter controls leg size, while the other controls leg spacing.

## About the paper

Paper: https://rkjones4.github.io/pdf/shapeMOD.pdf

Presented at [Siggraph 2021](https://s2021.siggraph.org/).

Project Page: https://rkjones4.github.io/shapeMOD.html

## Citations
```
  @article{jones2021shapeMOD,
      title={ShapeMOD: Macro Operation Discovery for 3D Shape Programs},
      author={Jones, R. Kenny and Charatan, David and Guerrero, Paul and Mitra, Niloy J. and Ritchie, Daniel},
      journal={ACM Transactions on Graphics (TOG), Siggraph 2021},
      volume={40},
      number={4},
      pages={Article 153},
      year={2021},
      publisher={ACM}
  }
```

## Running ShapeMOD

The ShapeMOD method is defined in ShapeMOD.py . Please check this file's comments for details on how the method is implemented.

We experiment with the ShapeMOD method on datasets of programs written in [ShapeAssembly](https://github.com/rkjones4/ShapeAssembly/). All ShapeAssembly specific logic lives in the SA_lang folder.

ShapeMOD.py consumes a config files and a dataset of programs. For instance using a config file SA_Lang/sa_config.py and a program dataset SA_lang/program_datasets/chair.prog you can run ShapeMOD with a command like:

```
python3 ShapeMOD.py SA_lang.sa_config SA_lang/program_datasets/chair.prog output
```

ShapeMOD will then run and write its results to the output folder. Note, to run this command please first download the chair.prog file following instructions in the SA_lang README.

## Understanding ShapeMOD Macros

All discovered libraries used in the paper's experiment can be found in SA_lang/tasks/dsls/ .

A discovered macro for ShapeMOD might look something like

```
('nfunc_0', (('Cuboid', 'f_var_0', 'f_bb_y * 1.0', 'f_var_0 * 1.0', 'b_var_0'), ('squeeze', 'i_bbox', 'i_bbox', 'c_bot', 'f_var_0 * 0.6', 'f_var_1'), ('reflect', 'c_X'))),
```

We could rewrite this macro in a more human readable format:

```
def macro_nfunc_0(f_var_0, b_var_0, f_var_1):
  cube0 = Cuboid(f_var_0, bb_y, f_var_0, b_var_0)
  Squeeze(bbox, boox, bot, 0.6 * f_var_0, f_var_1)
  Reflect(X)
```

Some notes on understanding ShapeAssembly macros:

Each variable has a type.

*i_{}* -> categorical variable that is part of the program structure (in ShapeAssembly these are cuboid indices, the bbox, or bounding box, is a special cuboid index that can treated as a constant). ret_{} are special variables in the category that correspond to returned objects created within the macro.

*c_{}* -> categorical variable that is part of the program parameterization (in ShapeAssembly these are the face of a Squeeze oprator or the axis of a Symmetry command)

*f_{}* -> float variables, always treated as program parameterization. bb_{} are special float variables that correspond to the bounding box dimensions, which are assumed as input for each program (because ShapeAssembly executes in a hierarchical fashion). f_1 is a special constant corresponding to 1.0

*b_{}* -> boolean variables, always treated as program parameterization. b_1 is True, b_0 is False

Each macro is represented wtihin the Function class, and calling printInfo() on the macro function will print out details about its sub-functions, free parameters and internal logic.

## Extending ShapeMOD to new domains

Extending ShapeMOD to a new domain will involve two steps, creating a dataset of programs and creating a config file for the new domain's language.

Dataset of programs are kept in .prog files. SA_lang/make_prog_dataset.py has the ShapeAssembly specific logic that creates the .prog files we use in our experiments.

A .prog file is a pickle dump of an array of ProgNode objects. Each ProgNode object gets a name and an array of OrderedProg objects. Each OrderedProg object is initialized with:

1. a signature corresponding to its line ordering. Note that programs that match on signature must be structural program matches (only differ in program parameterization)

2. the line attributes of that program (including structure and parameterization)

3. the return values of each line of the program

4. the values of any float parameters used in the program, sequentially flattened into a numpy array --> this is used to form clusters of related programs

The ProgNode and the OrderedProg classes are defined with ShapeMOD.py

To create a new config file for ShapeMOD please modifiy SA_lang/sa_config.py -- see the comments in this file to better understand how this configeration operates.

Please note the following assumptions for any new domain:

Assumption 1: All default parameters are assumed to be floats

Assumption 2: All base functions are assumed to be a single line

If you want to extend ShapeMOD to your domain, and one of these assumptions is broken, please raise a Github issue.

## Dependencies

Code was tested on Ubuntu 18.04 with pytorch 3.7. env.yml should contain a further list of all the conda packages/versions used to run the code.
