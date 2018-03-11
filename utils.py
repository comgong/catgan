from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function
#from __future__ import unicode_literals

import json
import numpy as np
import os
import sys
import tensorflow as tf

from collections import Counter, defaultdict
from scipy.misc import imsave
from six import iteritems
from subprocess import check_output
from time import strftime

def line_writer(od, newline=True):
    """
    Takes in an OrderedDict of (key, value) pairs
    and prints tab-separated with carriage return.
    """
    print_tuples = []
    for k, v in iteritems(od):
        if type(v) == str:
            print_tuples.append('{0} {1}'.format(k, v))
        else:
            v = float(v)
            print_tuples.append('{0}={1:.4f}'.format(k, v))
    #msg = "\r%s\n" if newline else "\r%s"
    msg = "%s\n" if newline else "\r%s"
    sys.stdout.write(msg % '\t'.join(print_tuples))
    sys.stdout.flush()
