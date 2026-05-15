"""Tombstone for the moved `saeco.sweeps.sweepable_config` package.

The contents of this module were extracted into the standalone
[`sweepable`](https://github.com/nqgl/sae_components/tree/master/sweepable)
package. Any import of `saeco.sweeps.sweepable_config[.X]` will fail at
import time so the broken caller is found immediately rather than producing
a misleading error later.

Migration:

    # Old:
    from saeco.sweeps.sweepable_config import SweepableConfig
    from saeco.sweeps.sweepable_config.sweepable_config import SweepableConfig
    from saeco.sweeps.sweepable_config.Swept import Swept

    # New:
    from sweepable import SweepableConfig, Swept
    # …or for less-common symbols:
    from sweepable.sweepable_config import SweepableConfig

`from saeco.sweeps import SweepableConfig` (and the other names re-exported
through `saeco.sweeps.__init__`) continues to work — those re-exports now
forward to `sweepable`.
"""

raise ImportError(
    "saeco.sweeps.sweepable_config has been extracted into the standalone "
    "`sweepable` package. Update your imports:\n"
    "    from sweepable import SweepableConfig, Swept, SweepVar, SweepExpression, Val\n"
    "Install with `pip install sweepable` (or `pip install -e sweepable` "
    "from the saeco repo root)."
)
