#!/usr/bin/env python3
"""
Test runner script for the PINN project.
Run all tests and generate a summary report.
"""

import os
import sys
import unittest

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def discover_and_run_tests():
    """Discover and run all tests in the tests directory"""

    # Get the tests directory
    test_dir = os.path.dirname(os.path.abspath(__file__))

    # Discover all test files
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern="test_*.py")

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            err_msg = traceback.split("AssertionError: ")[-1].split("\n")[0]
            print(f"- {test}: {err_msg}")

    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            err_msg = traceback.split("\n")[-2]
            print(f"- {test}: {err_msg}")

    if result.skipped:
        print("\nSKIPPED:")
        for test, reason in result.skipped:
            print(f"- {test}: {reason}")

    # Return success status
    return len(result.failures) == 0 and len(result.errors) == 0


def run_specific_test_file(test_file):
    """Run a specific test file"""
    if not test_file.startswith("test_") or not test_file.endswith(".py"):
        print(f"Error: {test_file} doesn't follow test file naming convention")
        return False

    # Import and run the specific test
    module_name = test_file[:-3]  # Remove .py extension
    try:
        module = __import__(module_name)
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(module)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return len(result.failures) == 0 and len(result.errors) == 0
    except ImportError as e:
        print(f"Error importing {module_name}: {e}")
        return False


def main():
    """Main test runner"""
    if len(sys.argv) > 1:
        # Run specific test file
        test_file = sys.argv[1]
        success = run_specific_test_file(test_file)
    else:
        # Run all tests
        success = discover_and_run_tests()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
