#!/usr/bin/env python3

import os
from context import medhok

from medhok import get_dataset as ds

ds.write_tf_records(split=False)
ds.write_tf_records(split=True)
