from traitlets import (Int, Integer, Float, Unicode, Enum, Long, List,
                       Bool, CRegExp, Dict, TraitError, observe,
                       CaselessStrEnum, TraitType)
from traitlets.config import boolean_flag as flag
import os

__all__ = ['Path', 'Int', 'Integer', 'Float', 'Unicode', 'Enum', 'Long', 'List',
           'Bool', 'CRegExp', 'Dict', 'flag', 'TraitError', 'observe',
           'CaselessStrEnum']


class Path(TraitType):
    def __init__(self, exists=None, directory_ok=True, file_ok=True):
        '''
        A path Trait for input/output files.

        Parameters
        ----------
        exists: boolean or None
            If True, path must exist, if False path must not exist

        directory_ok: boolean
            If False, path must not be a directory
        file_ok: boolean
            If False, path must not be a file
        '''
        super().__init__()
        self.exists = exists
        self.directory_ok = directory_ok
        self.file_ok = file_ok

    def validate(self, obj, value):

        if isinstance(value, str):
            value = os.path.abspath(value)
            if self.exists is not None:
                if os.path.exists(value) != self.exists:
                    raise TraitError('Path "{}" {} exist'.format(
                        value,
                        'does not' if self.exists else 'must'
                    ))
            if os.path.exists(value):
                if os.path.isdir(value) and not self.directory_ok:
                    raise TraitError(
                        f'Path "{value}" must not be a directory'
                    )
                if os.path.isfile(value) and not self.file_ok:
                    raise TraitError(
                        f'Path "{value}" must not be a file'
                    )

            return value

        return self.error(obj, value)
