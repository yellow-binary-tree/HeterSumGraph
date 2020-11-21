# -*- coding: utf-8 -*-

import logging
import sys

logger = logging.getLogger("Summarization logger")

formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')

console_handler = logging.StreamHandler(sys.stdout)
console_handler.formatter = formatter
console_handler.setLevel(logging.INFO)
# console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)
