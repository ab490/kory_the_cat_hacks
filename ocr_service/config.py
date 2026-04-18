import string

CHARSET = (
    string.ascii_uppercase
    + string.ascii_lowercase
    + string.digits
    + " "
    + r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
)
BLANK = 0

PAGE_SIZE = (540, 258)      # W, H
LINE_H, LINE_W = 32, 512
