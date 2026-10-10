# PhysX empty filter rows

Actual SDK observations show filter_count=0 and24 empty filter-path rows. This revision accepts exactly the shared-empty or24-empty-row representation, preserving sensor paths/count and native validity guards. Nonempty/ragged filters are rejected.

The shared probe/train entries migrate only the exact factory source pin. Dataset preflight from v5 is retained. All original functions of the entries are AST identical to v5. Actual shared World and training execution must still be verified.
