#!/usr/bin/env python
# -*- coding: utf-8 -*-


import unittest
from fractions import Fraction

import sys

sys.path.append("..")

from pts_generator import PtsGenerator


class PtsGeneratorTest(unittest.TestCase):
    def test_generate(self):
        gen = PtsGenerator(1.1, Fraction(1, 25))
        plist = gen.generate(2.09)
        self.assertEqual(25, len(plist))
        plist = gen.generate(2.1)
        self.assertEqual(1, len(plist))
        plist = gen.generate(3.1)
        self.assertEqual(25, len(plist))


if __name__ == "__main__":
    unittest.main()
