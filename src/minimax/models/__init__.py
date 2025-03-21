"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from .registration import register, make


from .maze import (
    GridWorldACStudentModel,
    GridWorldACTeacherModel,
)

from .overcooked import (
    ACStudentModel,
    ACStudentCriticModel,
    ACStudentActorModel,
    ACTeacherModel,
)

__all__ = [
    register,
    make,
    GridWorldACStudentModel,
    GridWorldACTeacherModel,
    ACStudentModel,
    ACStudentCriticModel,
    ACStudentActorModel,
    ACTeacherModel,
]
