## Data format
The csv files go in pair: surface and gradient. 
The filename has the 
identification of the dimension being considered.
For example, `gradidents_01.csv` has the gradient for dimension `0` and `1`. 
Same thing with surface.
 
- The surface is a 3D point cloud coordinations for drawing the loss surface.
 The format is as follow: `<theta_x>    <theta_y>   <loss>`.
- The gradients associated with the surface which has the gradient movements 
over time. In fact, each step has the same format with the surface, where it 
has the coordination of 2 `theta` and the associated loss values being 
optimized. 

## Suggestion: 
- Draw the surface first 
- Draw gradients on top of the surface by time step. Each line is one time 
step. 
