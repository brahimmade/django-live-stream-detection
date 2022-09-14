import os
from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]          # settings directory (root)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

HOST_IP = '192.168.1.7'     # paste your server ip address here
PORT1 = 9999                # port for first detection instance
PORT2 = 12345               # port for second detection instance


WRITE_IMAGE_INFO = True     # write detected objects info on the image
FONT_SCALE = 2
FONT_COLOR = (0, 0, 255)    # tupe BGR not RGB
FONT_THICHNESS = 1
