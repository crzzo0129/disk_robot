from __future__ import annotations

import sys

from scripts.distill_forward_student import main as distill_main


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    distill_main(["--phase-conditioned", *arguments])


if __name__ == "__main__":
    main()
