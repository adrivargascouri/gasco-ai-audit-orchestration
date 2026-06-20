"""Modular rule-based GASCO comparison pipeline.

This path is retained for parity testing against the legacy rule-based pipeline.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from modular.config.loader import ConfigLoader
from modular.orchestrator.pipeline import AuditScopePipeline


def main():
    """Run the modular audit scope pipeline."""
    # Load configuration from YAML
    config_path = Path(__file__).parent.parent / "modular" / "data" / "config.yaml"
    config = ConfigLoader.load_from_yaml(config_path)

    # Create and run pipeline
    pipeline = AuditScopePipeline(config)
    results = pipeline.run()

    # Print results
    pipeline.print_results(results)

    # Print export confirmation
    print("\n\nExported files:")
    for name, path in results["export_paths"].items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
