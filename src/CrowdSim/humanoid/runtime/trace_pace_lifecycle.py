"""Terminal release for one vectorized AMP articulation without contact views.

Reuses the verified isolated evaluator's SDK6 terminal release. This scope
supports flat/shared static cuboid worlds; it rejects sensors/native monitors
until their owner-specific release has been reviewed. Never masks process exit.
"""

from .social_evaluation_lifecycle import IsolatedTrackingLifecycle


class TracePaceLifecycle(IsolatedTrackingLifecycle):
    def _detach_world(self, world, producer):
        if world is None or producer is not None or world.scene_provider is not None:
            raise RuntimeError(
                "Require a complete generated-ground AMPWorld without native scene provider"
            )
        simulator = world.simulator
        scene, asset = simulator._scene, simulator._robot
        if (
            set(scene.articulations) != {"robot"}
            or scene.articulations["robot"] is not asset
            or scene.sensors
            or scene.rigid_objects
            or scene.rigid_object_collections
            or scene.deformable_objects
            or simulator._contact_sensor_map
            or simulator._object
            or simulator._projectile_objects
            or getattr(simulator, "pedestrian_contacts", None) is not None
        ):
            raise RuntimeError(
                "Additional native owners require an explicit terminal release"
            )
        self.release_asset(simulator._sim, (asset,), self.report)
        scene.articulations.clear()
        simulator._robot = simulator._scene = simulator._sim = (
            simulator._simulation_app
        ) = None
        world.simulator = None
        self.report["evaluation_world_detached"] = True
