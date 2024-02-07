# Interactive User Interface for Smocking Design \& Preview

<p align="center">
  <img align="center"  src="../figs/eg_ui.jpg" width="600">
</p>

We implement an interactive user interface in Blender as an add-on including the following functionalities:

- ***define a unit smocking pattern*** by creating a 2D grid and drawing stitching lines-
-  ***define a full smocking pattern*** by:
   - tiling the loaded unit smocking pattern (with user-defined repetition and shift of the unit pattern)
   - drawing stitching lines directly on a square or hexagonal grid to define the full pattern
- ***modify a full smocking pattern*** by
   - deforming the square grid into a radial grid (with user-defined radius)
   - adding margins to the pattern
   - combining it with another smocking pattern (along user-specified axis and space)
   - deleting/adding stitching lines from/to the pattern
- ***simulate the smocked pattern*** with intermediate steps including
   - extracting the smocked graph from the pattern
   - embedding the underlay and pleat subgraphs of the smocked graph
   - applying ARAP to compute the smocking design
- ***render the smocking design***
- ***run cloth simulator*** implemented in Blender on the fine-resolution smocking pattern
  
## Usage 
1. open our [add-on script](https://github.com/llorz/SmockingDesign/blob/main/python_implementation/utils/SmockingDesignAddOn.py) in Blender's scripting workspace and run it from there
2. compile cpp_smocking_solver.cpp (the compiled code for Mac with python 3.10 is provided)
3. follow our [supplematory video](https://youtu.be/vjnmbmO3zcg) to see some examples of using this UI
