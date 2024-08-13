# -*- coding: utf-8 -*-
import unittest
from unittest.mock import patch
import os
import sys
decquant_path = os.environ.get('DECOQUANT_PATH')

class TestSvdModuleImport(unittest.TestCase):
    def setUp(self):
        self.module_path = os.path.join(decquant_path, "external_modules/bdcsvd/build/lib.linux-x86_64-3.10")
        # self.module_path = os.path.join(decquant_path, "external_modules/bdcsvd/build/lib.linux-x86_64-3.8")

        sys.path.append(self.module_path)

    def tearDown(self):
        sys.path.remove(self.module_path)

    def test_svd_module_import(self):
        try:
            import svd_module
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import svd_module: {e}")
        print("Succeed loading svd_module")
    
if __name__ == '__main__':
    unittest.main()