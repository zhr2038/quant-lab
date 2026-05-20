"""Research workflows.

Import concrete workflow modules directly, for example
``quant_lab.research.bootstrap_gold``.  The package initializer intentionally
keeps imports lazy so risk, API, and export modules do not form circular
dependencies when they only need lightweight research metadata.
"""

__all__: list[str] = []
