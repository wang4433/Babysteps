"""BABYSTEPS Stage-4 — learned-latent track (see goal.md §"Stage 4").

This subpackage hosts only sim-free analysis code: feature extraction over
the demo-evidence fields of existing Stage-0 episode JSONs, sklearn linear
probes, and the per-task per-factor report builder.

Privileged-firewall invariant: every Stage-4 module here must consume only
DemoEvidence-shaped inputs (object_trajectory, contact_region_label,
final_state). It must never read execution.initial_intent (label leakage),
failure_packet, revision, retry, or any privileged SceneState field. The
firewall is what makes the recoverability number meaningful.
"""
