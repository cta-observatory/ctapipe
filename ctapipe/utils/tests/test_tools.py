import pytest
import traitlets
from traitlets import HasTraits
from traitlets import Int
# using this class as test input
from ctapipe.image.extractor import ImageExtractor


def test_enum_trait_default_is_right():
    # function under test
    from ctapipe.utils.tools import enum_trait

    with pytest.raises(ValueError):
        enum_trait(ImageExtractor, default='name_of_default_choice')


def test_enum_trait():
    # function under test
    from ctapipe.utils.tools import enum_trait

    trait = enum_trait(ImageExtractor, default='NeighborPeakWindowSum')
    assert isinstance(trait, traitlets.traitlets.CaselessStrEnum)


def test_enum_classes_with_traits():
    # function under test
    from ctapipe.utils.tools import classes_with_traits

    list_of_classes = classes_with_traits(ImageExtractor)
    assert list_of_classes  # should not be empty


def test_has_traits():
    # function under test
    from ctapipe.utils.tools import has_traits

    class WithoutTraits(HasTraits):
        pass

    class WithATrait(HasTraits):
        my_trait = Int()

    assert not has_traits(WithoutTraits)
    assert has_traits(WithATrait)
