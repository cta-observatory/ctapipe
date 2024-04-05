"""
This class is largely created as a more versatile alternative to the Linear ND
interpolator, allowing the user to pass the interpolator an array of classes and then
perform interpolation between the results of a member function of that class. Currently
the name of the target interpolation function is passed in a string at initialisation.
TODO:
- Figure out what to do when out of bounds of interpolation range
- Implement for higher dimensions
- Create function to append points to the interpolator
"""

import numpy as np
import numpy.ma as ma
from scipy.ndimage import map_coordinates
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

    def __init__(
        self,
        keys,
        values,
        function_name=None,
        remember_last=False,
        bounds=None,
        dtype=None,
    ):
        """
        Parameters
        ----------
        keys: ndarray
            Interpolation grid points
        values: ndarray
            Interpolation values
        function_name: str
            Name of class member function to call in the case we are interpolating
            between class predictions, for numpy arrays leave blank
        """
        self.keys = keys
        if dtype:
            self.values = np.array(values, dtype=dtype)
        else:
            self.values = np.array(values)

        self._n_dimensions = len(self.keys[0])

        # create an object with triangulation
        self._tri = Delaunay(self.keys)
        self._function_name = function_name

        # OK this code is horrid and will need fixing
        self._numpy_input = (
            isinstance(self.values[0], np.ndarray)
            or issubclass(type(self.values[0]), np.floating)
            or issubclass(type(self.values[0]), np.integer)
        )

        if self._numpy_input is False and function_name is None:
            self._function_name = "__call__"

        self._remember = remember_last

        if bounds is not None:
            bounds = np.array(bounds, dtype=dtype)
            self._bounds = bounds

            # Calculate the scaling factor to convert from bin number to real
            # coordinates for all axes
            scale = []
            table_shape = self.values[0].shape
            for i in range(bounds.shape[0]):
                scale_dimemsion = bounds[i][1] - bounds[i][0]
                scale_dimemsion = scale_dimemsion / float(table_shape[i] - 1)
                scale.append(scale_dimemsion)
            self.scale = np.array(scale, dtype=dtype)

        self._previous_v = None
        self._previous_m = None
        self._previous_shape = None
        self._previous_hull = None
        self._previous_points = None
        self.reset()

    def reset(self):
        """
        Function used to reset some class values stored after previous event,
        also used as their initialisation in the init function
        """
        self._previous_v = None
        self._previous_m = None
        self._previous_shape = None
        self._previous_hull = None
        self._previous_points = None

    def __call__(self, points, eval_points=None):
        # Convert to a numpy array here in case we get a list
        points = np.array(points, dtype=np.float32)
        if eval_points is not None:
            eval_points = eval_points.astype(np.float32)

        if len(points.shape) == 1:
            points = np.array([points])

        # First find simplexes that contain interpolated points

        if self._remember and self._previous_v is not None:
            # We have a few different options here in the case that the points we get are similar
            # to the last set given

            # Our first check is if the interpolation points are exactly the same
            # in this case just use the previous set of vertices
            if np.all(points == self._previous_points):
                v = self._previous_v
                m = self._previous_m
            else:
                # If not we can check if our point set exists within the previous
                # simplax
                previous_keys = self.keys[self._previous_v.ravel()]
                self._previous_points = points

                if self._previous_hull is None:
                    hull = Delaunay(previous_keys)
                    self._previous_hull = hull
                else:
                    hull = self._previous_hull

                if np.all(eval_points is not None):
                    shape_check = eval_points.shape == self._previous_shape
                else:
                    shape_check = True

                # If it does then we can use this simplex
                if np.all(hull.find_simplex(points) >= 0) and shape_check:
                    v = self._previous_v
                    m = self._previous_m
                # If not we have to search through our point space
                else:
                    s = self._tri.find_simplex(points)
                    v = self._tri.simplices[s]
                    m = self._tri.transform[s]
                    self._previous_v = v
                    self._previous_m = m
                    self._previous_hull = None

                    if np.all(eval_points is not None):
                        self._previous_shape = eval_points.shape
        # If remember last is disabled we search our point space for every attempt
        else:
            s = self._tri.find_simplex(points)
            # get the vertices for each simplex
            v = self._tri.simplices[s]
            # get transform matrices for each simplex
            m = self._tri.transform[s]
            self._previous_v = v
            self._previous_m = m
            self._previous_points = points
            if np.all(eval_points is not None):
                self._previous_shape = eval_points.shape

        # Here comes some serious numpy magic, it could be done with a loop but would
        # be pretty inefficient I had to rip this from stack overflow - RDP
        # For each interpolated point, take the the transform matrix and multiply it by
        # the vector p-r, where r=m[:,n,:] is one of the simplex vertices to which
        # the matrix m is related to
        b = np.einsum(
            "ijk,ik->ij",
            m[:, : self._n_dimensions, : self._n_dimensions],
            points - m[:, self._n_dimensions, :],
        )

        # Use the above array to get the weights for the vertices; `b` contains an
        # n-dimensional vector with weights for all but the last vertices of the simplex
        # (note that for n-D grid, each simplex consists of n+1 vertices);
        # the remaining weight for the last vertex can be copmuted from
        # the condition that sum of weights must be equal to 1
        w = np.c_[b, 1 - b.sum(axis=1)]

        if self._numpy_input:
            if eval_points is None:
                selected_points = self.values[v]
            else:
                selected_points = self._numpy_interpolation(v, eval_points)
        else:
            selected_points = self._call_class_function(v, eval_points)

        # Multiply point values by weight
        p_values = np.einsum("ij...,ij...->i...", selected_points, w)

        return p_values

    def _call_class_function(self, point_num, eval_points):
        """
        Function to loop over class function and return array of outputs
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

        three_dim = False
        if len(eval_points.shape) > 2:
            first_index = np.arange(point_num.shape[0])[..., np.newaxis] * np.ones_like(
                point_num
            )
            first_index = first_index.ravel()
            three_dim = True

        num = 0
        for pt in point_num.ravel():
            cls = self.values[pt]
            cls_function = getattr(cls, self._function_name)
            pt = eval_points
            if three_dim:
                pt = eval_points[first_index[num]]

            outputs.append(cls_function(pt))
            num += 1

        outputs = np.array(outputs)
        new_shape = (*shape, *outputs.shape[1:])
        outputs = outputs.reshape(new_shape)

        return outputs

    def _numpy_interpolation(self, point_num, eval_points):
        """
        Perform 2D interpolation of numpy array

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

        # Check if our array is masked and remember its shape
        is_masked = ma.is_masked(eval_points)
        shape = point_num.shape

        # Get the list of templates that we want to interpolate between
        ev_shape = eval_points.shape
        vals = self.values[point_num.ravel()]

        # Scale the template x and y axes to convert into bin coordinates
        scaled_points = eval_points.T
        scaled_points[0] = (scaled_points[0] - self._bounds[0][0]) / self.scale[0]
        scaled_points[1] = (scaled_points[1] - self._bounds[1][0]) / self.scale[1]
        eval_points = scaled_points.T

        # This gets a bit ugly now but the general logic is...
        # for each point in the phase space repeat the x-y points by the number
        # of points which define the simplex
        eval_points = np.repeat(eval_points, shape[1], axis=0)
        it = np.arange(eval_points.shape[0])
        it = np.repeat(it, eval_points.shape[1], axis=0)

        eval_points = eval_points.reshape(
            eval_points.shape[0] * eval_points.shape[1], eval_points.shape[-1]
        )

        # Make a mask to be sure masked array values are not included
        scaled_points = eval_points.T
        if is_masked:
            mask = np.invert(ma.getmask(scaled_points[0]))
        else:
            mask = np.zeros_like(scaled_points[0], dtype=bool)

        it = ma.masked_array(it, mask)

        if not is_masked:
            mask = ~mask

        # Then stack up all the templates and do a 3d interpolation of points in all templates
        scaled_points = np.vstack((it, scaled_points))
        scaled_points = scaled_points.astype(self.values.dtype)

        output = np.zeros(scaled_points.T.shape[:-1])
        output = map_coordinates(vals, scaled_points, order=1, output=self.values.dtype)

        new_shape = (*shape, ev_shape[-2])
        output = output.reshape(new_shape)

        # Return a masked array of interpolated values
        return ma.masked_array(output, mask=mask, dtype=self.values.dtype)
