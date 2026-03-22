"""
ComfyUI SAM3 Better Segmentation — SAM3BS node package.
"""

import sys
from pathlib import Path

current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

print("\n" + "=" * 70)
print("[SAM3BS] ComfyUI SAM3 Better Segmentation - Loading")
print("=" * 70)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from .nodes import NODE_CLASS_MAPPINGS as NODES
    from .nodes import NODE_DISPLAY_NAME_MAPPINGS as NAMES

    NODE_CLASS_MAPPINGS.update(NODES)
    NODE_DISPLAY_NAME_MAPPINGS.update(NAMES)

    print(f"[SAM3BS] Registered {len(NODE_CLASS_MAPPINGS)} nodes:")
    for node_id, cls in NODE_CLASS_MAPPINGS.items():
        display_name = NODE_DISPLAY_NAME_MAPPINGS.get(node_id, node_id)
        print(f"[SAM3BS]  - {display_name} ({node_id})")
    print("=" * 70)

except Exception as e:
    import traceback

    print(f"[SAM3BS] ERROR while loading nodes: {e}")
    traceback.print_exc()
    print("[SAM3BS] Nodes will be unavailable until this is fixed.")
    print("=" * 70)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
