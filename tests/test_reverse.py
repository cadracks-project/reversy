#!/usr/bin/env python
# coding: utf-8

r"""Test reverse"""

from os.path import join, dirname

from reversy.reversy import reverse


def test_reverse():
    step_file = "../step/ASM0001_ASM_1_ASM.stp"  # OCC compound
    step_path = join(dirname(__file__), step_file)
    a1 = reverse(step_path, view=False)
    assert a1 is not None
