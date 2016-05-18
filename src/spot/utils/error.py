class SpotError(Exception):
    def __init__(self, message, cause=None):
        super(SpotError, self).__init__(message + (u', caused by ' + repr(cause) if cause else ''))
        self.cause = cause
