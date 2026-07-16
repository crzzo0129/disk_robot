from __future__ import annotations

import sys

from scripts.dagger_forward_student import main as dagger_main


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    dagger_main(["--require-phase-conditioned", *arguments])


if __name__ == "__main__":
    main()
