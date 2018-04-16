"""
TODO:
- Figure out what to do when out of bounds
"""

import numpy as np
from scipy.spatial import Delaunay


class UnstructuredInterpolator:
    """
    This class performs linear interpolation between an unstructured set of data
    points. However the class is expanded such that it can interpolate between the
    values returned from class member functions. The primary use case of this being to
    interpolate between the predictions of a set of machine learning algorithms or
    regular grid interpolators.

    In the case that a numpy array is passed as the interpolation values this class will
    behave exactly the same as the scipy LinearNDInterpolator
    """
    def __init__(self, interpolation_points, function_name=None):
        """
        Parameters
        ----------
        interpolation_points: dict
            Dictionary of interpolation points (stored as key) and values
        function_name: str
            Name of class member function to call in the case we are interpolating
            between class predictions, for numpy arrays leave blank
        """

        self.keys = np.array(list(interpolation_points.keys()))
        self.values = np.array(list(interpolation_points.values()))

        self.num_dimensions = len(self.keys[0])

        # create an object with triangulation
        self.tri = Delaunay(self.keys)
        self.function_name = function_name

        # OK this code is horrid and will need fixing
        self.numpy_input = isinstance(self.values[0], np.ndarray) or \
                           issubclass(type(self.values[0]), np.float) or \
                           issubclass(type(self.values[0]), np.int)

        if self.numpy_input == False and function_name == None:
            self.function_name = "__call__"

        return None

    def __call__(self, points, eval_points=None):

        if self.numpy_input == False and eval_points==None:
            raise ValueError("Non numpy object provided without with emtpy eval_points")

        # Convert to a numpy array here incase we get a list
        points = np.array(points)

        if len(points.shape) == 1:
            points = np.array([points])

        # find simplexes that contain interpolated points
        s = self.tri.find_simplex(points)
        # get the vertices for each simplex
        v = self.tri.vertices[s]
        # get transform matrices for each simplex (see explanation bellow)
        m = self.tri.transform[s]

        # Here comes some serious numpy magic, it could be done with a loop but would
        # be pretty inefficient I had to rip this from stack overflow - RDP
        # For each interpolated point, take the the transform matrix and multiply it by
        # the vector p-r, where r=m[:,n,:] is one of the simplex vertices to which
        # the matrix m is related to
        b = np.einsum('ijk,ik->ij', m[:, :self.num_dimensions, :self.num_dimensions],
                      points - m[:,self.num_dimensions, :])

        # Use the above array to get the weights for the vertices; `b` contains an
        # n-dimensional vector with weights for all but the last vertices of the simplex
        # (note that for n-D grid, each simplex consists of n+1 vertices);
        # the remaining weight for the last vertex can be copmuted from
        # the condition that sum of weights must be equal to 1
        w = np.c_[b, 1 - b.sum(axis=1)]

        if self.numpy_input:
            selected_points = self.values[v]
        else:
            selected_points = self._call_class_function(v, eval_points)

        # Multiply point values by weight
        p_values = np.einsum('ij...,ij...->i...', selected_points, w)
        return p_values

    def _call_class_function(self, point_num, eval_points):
        """

        Parameters
        ----------
        point_num: int
            Index of class position in values list
        eval_points: ndarray
            Inputs used to evaluate class member function

        Returns
        -------
        ndarray: output from member function
        """

        outputs = list()
        shape = point_num.shape
        for pt in point_num.ravel():
            cls = self.values[pt]
            cls_function = getattr(cls, self.function_name)
            outputs.append(cls_function(eval_points))
        outputs = np.array(outputs)
        outputs = outputs.reshape(shape)

        return outputs