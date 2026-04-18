# Shared constants: charset + image sizes.

import string

# CTC needs a blank token at index 0 — real chars start at 1.
CHARSET = (
    string.ascii_uppercase
    + string.ascii_lowercase
    + string.digits
    + " "
    + r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
)
BLANK = 0

# Page input to the detector, line crop input to the recognizer.
PAGE_SIZE = (540, 258)      # W, H — matches SimulatedNoisyOffice patches
LINE_H, LINE_W = 32, 512    # recognizer accepts 32-px tall line crops
