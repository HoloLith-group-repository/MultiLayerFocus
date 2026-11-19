"""PyInstaller entry point for MultiLayerFocus (GUI, or batch mode with --run)."""
import sys

if __name__ == '__main__':
    if '--run' in sys.argv:
        from multilayerfocus.cli import main as batch
        raise SystemExit(batch(sys.argv[1:]))
    from multilayerfocus.app import main
    raise SystemExit(main())
