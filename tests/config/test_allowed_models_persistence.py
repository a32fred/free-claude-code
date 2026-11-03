"""Admin updates must preserve a manually configured route restriction."""

import json

from free_claude_code.config.admin.persistence import prepare_admin_update
from free_claude_code.config.loader import ManagedConfigStore


def test_unrelated_admin_update_keeps_allowed_models():
    store = ManagedConfigStore()
    store.initialize()
    refs = ["nvidia_nim/approved"]
    store.commit(dict(store.read().managed) | {"FCC_ALLOWED_MODELS": json.dumps(refs)})
    snapshot = store.read()
    update = prepare_admin_update(
        {"MODEL": "nvidia_nim/approved"}, snapshot, snapshot.settings,
        revision_key=b"test-key",
    )
    assert update.valid
    store.commit(update.target_values)
    assert store.read().settings.allowed_models == tuple(refs)
