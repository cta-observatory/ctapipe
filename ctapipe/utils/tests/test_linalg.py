from ..linalg import rotation_matrix_2d
from numpy import dot, allclose, identity


def test_rotation_matrix_2d():

    # test that 360 rotation is back to the identity:
    assert allclose(rotation_matrix_2d('360d'), identity(2))

    # test that a vector can be rotated correcly:
    vec = [1, 0]
    mat = rotation_matrix_2d('90d')
    vecp = dot(vec, mat)
    assert allclose(vecp, [0, -1]), 'vector rotation is wrong'

    # test that the rotation is Hermitian

    m = rotation_matrix_2d('25d')
    assert allclose(dot(m, m.T), identity(2)), "rotation should be Hermetian"
