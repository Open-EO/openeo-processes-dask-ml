from openeo_processes_dask.process_implementations.exceptions import OpenEOException


class LabelDoesNotExist(OpenEOException):
    pass


class ExpressionEvaluationException(OpenEOException):
    pass


class BandNotFoundException(OpenEOException):
    pass


class ReferenceSystemNotFound(OpenEOException):
    pass


class SLURMException(Exception):
    pass
