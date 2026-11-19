import sys
if '--run' in sys.argv:
    from .cli import main
    raise SystemExit(main(sys.argv[1:]))
from .app import main
raise SystemExit(main())
