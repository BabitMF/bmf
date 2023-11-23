import os, time
from fractions import Fraction
import logging
from bmf.lib._bmf import engine
import bmf.lib._bmf as _bmf

## @ingroup pyAPI
## @defgroup pyAPIVer version
###@{
# BMF Version
###@}


class LogLevel:
    VERBOSE = logging.DEBUG - 1
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    FATAL = logging.CRITICAL
    DISABLE = logging.CRITICAL + 1


def get_log_level():
    if 'BMF_LOG_LEVEL' in os.environ:
        if os.environ['BMF_LOG_LEVEL'] == 'DEBUG':
            return LogLevel.DEBUG
        elif os.environ['BMF_LOG_LEVEL'] == 'INFO':
            return LogLevel.INFO
        elif os.environ['BMF_LOG_LEVEL'] == 'WARNING':
            return LogLevel.WARNING
        elif os.environ['BMF_LOG_LEVEL'] == 'ERROR':
            return LogLevel.ERROR
        elif os.environ['BMF_LOG_LEVEL'] == 'FATAL':
            return LogLevel.FATAL
        elif os.environ['BMF_LOG_LEVEL'] == 'DISABLE':
            return LogLevel.DISABLE


class Log:
    log_level = get_log_level()
    start_time = -1
    logging.basicConfig()
    logger = logging.getLogger('main')
    if log_level is None:
        log_level = LogLevel.INFO
    else:
        logger.setLevel(log_level)  # Set only if BMF_LOG_LEVEL is set

    @staticmethod
    def get_curr_time():
        if Log.start_time < 0:
            Log.start_time = time.time()
            return 0
        else:
            return time.time() - Log.start_time

    @staticmethod
    def set_log_level(l):
        Log.log_level = l
        Log.logger.setLevel(Log.log_level)

    @staticmethod
    def log(ll, *a):
        if ll >= Log.log_level:
            str_a = []
            for item in a:
                str_a.append(str(item))
            message = '%f -- %s' % (Log.get_curr_time(), ' '.join(str_a))
            Log.logger.log(ll, message)

    @staticmethod
    def log_node(ll, node_id, *a):
        if ll >= Log.log_level:
            str_a = []
            for item in a:
                str_a.append(str(item))
            message = '%f -- (%d) -- %s' % (Log.get_curr_time(), node_id,
                                            ' '.join(str_a))
            Log.logger.log(ll, message)


av_time_base = Fraction(1, 1000000)


def scale_av_pts(pts, time_base_1, time_base_2):
    if pts is not None:
        return round(float(pts * time_base_1 / time_base_2))
    return None


## @ingroup pyAPIVer
###@{
#  @brief get bmf version
#  @return version string
def get_version():
    ###@}
    return _bmf.get_version()


## @ingroup pyAPIVer
###@{
#  @brief get commit id
#  @return commit (short) string
def get_commit():
    ###@}
    return _bmf.get_commit()


## @ingroup pyAPIVer
###@{
#  @brief change dmp file output path
def change_dmp_path(path):
    ###@}
    return engine.change_dmp_path(path)
