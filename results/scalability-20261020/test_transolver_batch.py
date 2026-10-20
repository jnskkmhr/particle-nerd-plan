# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batched environments must retain independent particle predictions."""

import torch

from particle_nerd.transolver.model import ParticleTransolver


def test_batch_matches_independent_environments_and_isolates_perturbations():
    torch.manual_seed(9)
    model = ParticleTransolver(hidden=32, heads=4, slices=8, layers=2).eval()
    torch.nn.init.normal_(model.decoder[-1].weight, std=0.1)
    particles = torch.randn(3, 21, 25)
    colliders = torch.randn(3, 21, 2, 8)
    weights = torch.rand(3, 21, 2)
    with torch.no_grad():
        independent = torch.stack([model(particles[e], colliders[e], weights[e]) for e in range(3)])
        batched = model(particles, colliders, weights)
        torch.testing.assert_close(batched, independent, atol=1e-7, rtol=1e-5)
        particles[1] += 2.0
        changed = model(particles, colliders, weights)
        torch.testing.assert_close(changed[[0, 2]], batched[[0, 2]], atol=0, rtol=0)
        assert not torch.allclose(changed[1], batched[1])
