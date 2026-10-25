"""Shared robot contact geometry and physics-material binding.

The policy is available offline. USD imports stay inside the runtime functions.
"""

from __future__ import annotations

import math


FINGER_PREFIXES = tuple(side+finger for side in ("l_", "r_")
                        for finger in ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky"))


def collision_policy(config, model, extra_decomposition_links=()):
    """Resolve the effective approximations, including legacy task overrides."""
    extra = set(extra_decomposition_links)
    unknown = extra-set(model.collidable_links)
    if unknown:
        raise ValueError("Unknown collision decomposition links: "+", ".join(sorted(unknown)))
    result = {}
    for name in model.collidable_links:
        if name in ("l_mount", "r_mount") or name in extra:
            result[name] = {"approximation": "convexDecomposition", "max_hulls": 64}
        elif name == "base_link" or name.endswith(("_force_sensor", "_camera_bracket")):
            # Retain the effective geometry of the successful physical task.
            # Earlier scene code intended decomposition here but skipped the
            # imported Xform mesh roots; enabling it is a separate model change.
            result[name] = {"approximation": "convexHull", "max_hulls": 64}
        elif name.startswith(FINGER_PREFIXES):
            result[name] = {"approximation": config["robot"]["collision"]["finger_approximation"], "max_hulls": 64}
    return result


def _link_name(path, robot_path):
    return str(path).removeprefix(robot_path+"/").split("/")[0]


def configure_robot_collisions(stage, robot_path, model, config, extra_decomposition_links=()):
    """Apply settings to actual collision shapes, including imported instances."""
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema

    policy = collision_policy(config, model, extra_decomposition_links)
    paths = [str(prim.GetPath()) for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_path), Usd.TraverseInstanceProxies())
             if prim.HasAPI(UsdPhysics.CollisionAPI)]
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        while prim.IsInstanceProxy():
            parent = prim.GetParent()
            while parent.IsInstanceProxy():
                parent = parent.GetParent()
            parent.SetInstanceable(False)
            prim = stage.GetPrimAtPath(path)
        link = _link_name(path, robot_path)
        collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        collision.CreateContactOffsetAttr(config["rail"]["contact_offset_m"] if link.startswith(("l_", "r_")) else .002)
        collision.CreateRestOffsetAttr(config["rail"]["rest_offset_m"])
        settings = policy.get(link)
        # Imported mesh colliders can be Xform roots with MeshCollisionAPI;
        # their node type alone does not identify the collision geometry.
        if settings and (prim.IsA(UsdGeom.Mesh) or prim.HasAPI(UsdPhysics.MeshCollisionAPI)):
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(settings["approximation"])
            if settings["approximation"] == "convexDecomposition":
                decomposition = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
                decomposition.CreateMaxConvexHullsAttr(settings["max_hulls"])
                decomposition.CreateHullVertexLimitAttr(64)
                decomposition.CreateVoxelResolutionAttr(500000)
    material = UsdShade.Material.Get(stage, "/World/Materials/Fingers")
    if not material:
        raise RuntimeError("Scene finger physics material is missing")
    for name in model.collidable_links:
        if name.startswith(("l_", "r_")):
            link = stage.GetPrimAtPath(robot_path+"/"+name)
            UsdShade.MaterialBindingAPI.Apply(link).Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                                                       materialPurpose="physics")
    return inspect_robot_collisions(stage, robot_path, model, config, extra_decomposition_links)


def inspect_robot_collisions(stage, robot_path, model, config, extra_decomposition_links=()):
    """Verify effective bindings and offsets rather than their intended values."""
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema

    policy = collision_policy(config, model, extra_decomposition_links)
    bound_links = sorted(name for name in model.collidable_links if name.startswith(("l_", "r_")))
    colliders = []
    observed = set()
    for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_path), Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        path = str(prim.GetPath()); link = _link_name(path, robot_path); observed.add(link)
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
        physics = UsdPhysics.MaterialAPI(material.GetPrim()) if material else None
        collision = PhysxSchema.PhysxCollisionAPI(prim)
        record = {"path": path, "type": prim.GetTypeName(),
                  "approximation": UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get(),
                  "schemas": prim.GetAppliedSchemas(), "material": str(material.GetPath()) if material else None,
                  "static_friction": physics.GetStaticFrictionAttr().Get() if physics else None,
                  "dynamic_friction": physics.GetDynamicFrictionAttr().Get() if physics else None,
                  "contact_offset_m": collision.GetContactOffsetAttr().Get(),
                  "rest_offset_m": collision.GetRestOffsetAttr().Get(), "instance_proxy": prim.IsInstanceProxy()}
        for key, expected in (("contact_offset_m", config["rail"]["contact_offset_m"] if link.startswith(("l_", "r_")) else .002),
                              ("rest_offset_m", config["rail"]["rest_offset_m"])):
            if record[key] is None or not math.isclose(record[key], expected, rel_tol=1e-6, abs_tol=1e-9):
                raise RuntimeError(f"Robot {key} did not bind to {path}")
        if link in bound_links:
            if physics is None or not all(value is not None and math.isclose(value, config["rail"]["finger_friction"], abs_tol=1e-6)
                                          for value in (record["static_friction"], record["dynamic_friction"])):
                raise RuntimeError("Hand friction did not bind to "+path)
        if (link in policy and (prim.IsA(UsdGeom.Mesh) or prim.HasAPI(UsdPhysics.MeshCollisionAPI))
                and record["approximation"] != policy[link]["approximation"]):
            raise RuntimeError("Robot collision approximation did not bind to "+path)
        colliders.append(record)
    # The URDF importer replaces spaces in link names with underscores.
    missing = {name.replace(" ", "_") for name in model.collidable_links}-observed
    if missing:
        raise RuntimeError("Robot collision shapes are missing: "+", ".join(sorted(missing)))
    return {"validated": True, "finger_approximation": config["robot"]["collision"]["finger_approximation"],
            "effective_decomposition_links": sorted(name for name, item in policy.items() if item["approximation"] == "convexDecomposition"),
            "colliders": colliders,
            "hand_material_binding": {"scope": "scene defaults and explicit overrides", "coefficient": config["rail"]["finger_friction"],
                                      "static_and_dynamic": True, "validated": True, "bound_links": bound_links}}
