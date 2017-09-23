Code for Gradient Domain VCM

related folders:
# vcm implementation
src/integrators/vcm
# gdvcm implementation
src/integrators/gdvcm

Currently, gdvcm does not fully support materials with both specular and diffuse components.
The current experimental implementation is not memory efficient. So sufficient memory (~16GB for 1kx1k image) is needed in order to have timing performance reported in the paper.
