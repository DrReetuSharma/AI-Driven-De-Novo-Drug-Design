# test_main_module.py

import unittest
from mymodule import main_function, MyClass

class TestMainFunction(unittest.TestCase):
    def test_main_function(self):
        # Test main_function with different inputs
        self.assertEqual(main_function(1), 2)
        self.assertEqual(main_function(0), 1)
        self.assertEqual(main_function(-1), 0)

class TestMyClass(unittest.TestCase):
    def test_myclass_method(self):
        # Test MyClass method with different inputs
        obj = MyClass()
        self.assertEqual(obj.method(1), 2)
        self.assertEqual(obj.method(0), 1)
        self.assertEqual(obj.method(-1), 0)

if __name__ == '__main__':
    unittest.main()
