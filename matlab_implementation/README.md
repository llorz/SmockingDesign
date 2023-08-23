We also provide a matlab implementation for smocking design preview.
## Main functions
```solve_smocking_design.m```: takes a smocking pattern as input, outputs the simulated smocked design as a 3D mesh. It has two main components:
1. ```embed_smocked_graph_clean.m```: embed the smocked graph that is extracted from the input smocking pattern (note, the ```SG.compute_max_edge_length``` is the embedding distance constraint $d_{i,j}$ defined in Eq.(4) Fig.10 in the paper). 
2. ```arap_simulate_smocked_design.m```: use the embedded smocked graph to guide a finer reprentation of the fabric using arap

## Experiments
- ```eg1_arrow.m```: shows an example of running our algorithm for the arrow pattern
- ```eg2_smocked_graph.m```: shows how the smocked graph before and after embedding look like
