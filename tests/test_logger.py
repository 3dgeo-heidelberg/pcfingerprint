from pcfingerprint.logger import *

import logging
import os
import pytest


def test_set_loggerfile(tmp_path):
    filename = os.path.join(tmp_path, "test.log")
    set_pcfingerprint_logfile(filename)
    assert not os.path.exists(filename)
    logging.getLogger("pcfingerprint").info("Some log message")
    assert os.stat(filename).st_size > 0
