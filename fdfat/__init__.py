import platform

__version__ = '0.1.8'

LOGGING_NAME = 'landmark'
VERBOSE = 'true'
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ['Darwin', 'Linux', 'Windows'])  # environment booleans
TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'
