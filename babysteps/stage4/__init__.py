"""BABYSTEPS Stage-4 / Stage-5 — learned-latent track.

Stage 4 (complete): handcrafted 20-dim demo-evidence features → IntentHead
→ ReviseHead → centroid decode. Proved the slot-local interface works.

Stage 5 (active, ICLR target): replaces the handcrafted input with frozen
vision-encoder features (DINOv2/R3M on demo RGB frames). Adds VLM-based
attribution and world-model counterfactual verification. See goal.md
§"Stage 5" for the spec.

Privileged-firewall invariant (carries over): encoder-side modules
(features.py, future vision_features.py) must consume only
DemoEvidence-shaped inputs. They must never read execution.initial_intent
(label leakage), failure_packet, revision, retry, or any privileged
SceneState field. The firewall is what makes the recoverability number
meaningful — both for the Stage-4 handcrafted probe and the Stage-5
vision-grounded probe.
"""
