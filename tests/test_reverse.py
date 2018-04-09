#!/usr/bin/env python
# coding: utf-8

r"""Test reverse"""

from reversy.reversy import reverse


def test_reverse():
    filename = "../step/ASM0001_ASM_1_ASM.stp"  # OCC compound
    a1 = reverse(filename, view=False)
    assert a1 is not None