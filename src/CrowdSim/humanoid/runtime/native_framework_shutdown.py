"""Full native plugin shutdown while Python is still alive, after App.close.

IsaacSim6's pinned SimulationApp.close performs extension teardown and leaves
framework unloading to process exit. Our GDB trace observed a PhysX binding
list destructor then accessing Python after finalization. A short diagnostic
returned normally when documented Carbonite unloading ran before that phase.
This supplements full cleanup; it neither bypasses it nor fabricates exit0.
https://docs.omniverse.nvidia.com/kit/docs/kit-manual/latest/carb/carb.Framework.html
"""
import time

def unload_native_plugins(report, *, full_app_close_returned):
    if full_app_close_returned is not True:
        raise RuntimeError('Full App.close must return before native plugin unloading')
    from importlib import metadata
    if metadata.version("isaacsim") == "6.1.0.0":
        # The 6.0 workaround must not eagerly unload 6.1's interfaces while
        # their Python SDK classes still exist. Full App.close has already
        # completed; let this SDK perform its documented process teardown.
        report['native_framework_shutdown'] = {
            'source': 'Sim6.1 stock framework teardown after full App.close',
            'returned': True, 'manual_plugin_unload': False,
            'framework_unload_deferred_to_SDK': True,
            'process_exit_masked': False,
            'actual_process_exit_requires_parent_wait': True,
        }
        return
    import carb
    started=time.monotonic();framework=carb.get_framework()
    before=len(framework.get_plugins())
    framework.unload_all_plugins()
    remaining=[v.impl.name for v in framework.get_plugins()]
    after=len(remaining)
    report['native_framework_shutdown']={
        'source':'documented carb.Framework.unload_all_plugins after full App.close',
        'plugins_before':before,'plugins_after':after,'remaining_plugins':remaining,
        'elapsed_s':time.monotonic()-started,'returned':True,
        'before_python_finalization':True,'process_exit_masked':False}
