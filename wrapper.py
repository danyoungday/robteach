import numpy as np

from robosuite.models.objects import BoxObject
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.wrappers import Wrapper


class DictCurriculumWrapper(Wrapper):
    """
    Gym wrapper that allows updating the curriculum of the env online.
    When apply_curriculum is called, changes the sampler that reset() uses to sample the cube's spawn position and
    orientation according to the curriculum dict.
    """
    def create_placement_initializer(self, xrange, yrange, rotation):
        """
        From Robosuite Lift: creates the placement initializer for the cube object.
        """
        table_offset = np.array((0, 0, 0.8))

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        cube = BoxObject(
            name="cube",
            size_min=[0.020, 0.020, 0.020],  # [0.015, 0.015, 0.015],
            size_max=[0.022, 0.022, 0.022],  # [0.018, 0.018, 0.018])
            rgba=[1, 0, 0, 1],
            material=redwood
        )

        placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=cube,
            x_range=xrange,
            y_range=yrange,
            rotation=rotation,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=table_offset,
            z_offset=0.01
        )

        return placement_initializer

    def update_curriculum(self, curriculum_dict: dict):
        """
        Updates the curriculum of the env by changing the placement sampler to a one defined by the curriculum dict.
        """
        xrange = (float(curriculum_dict["cube_x_range"][0]), float(curriculum_dict["cube_x_range"][1]))
        yrange = (float(curriculum_dict["cube_y_range"][0]), float(curriculum_dict["cube_y_range"][1]))
        rotation = (float(curriculum_dict["cube_rotation_range"][0]), float(curriculum_dict["cube_rotation_range"][1]))
        placement_initializer = self.create_placement_initializer(xrange, yrange, rotation)

        # Update the underlying env's placement initializer
        self.env.placement_initializer = placement_initializer


class ResetCodeWrapper(Wrapper):
    """
    Gym wrapper that updates the reset code of an environment online so that the curriculum can change the reset
    behavior of the env. Takes in arbitrary code as a string and executes it when the underlying env calls reset.
    """
    pass