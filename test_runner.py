"""
Async test runner for Discord AI Assistant Bot unit tests

Run this file to execute all tests with proper async support.
"""

import asyncio
import unittest
import sys
from test_my_discord_ai_bot import *

async def run_async_tests():
    """Run all async tests."""
    # Get all test classes
    test_classes = [
        TestSystemPromptLoading,
        TestBotMentionHandling,
        TestLLMIntegration,
        TestMessageHandling,
        TestConfiguration,
        TestIntegration
    ]

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    return result.wasSuccessful()

def main():
    """Main function to run all tests."""
    print("Running Discord AI Assistant Bot Unit Tests")
    print("=" * 50)

    # Run async tests
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        success = loop.run_until_complete(run_async_tests())

        if success:
            print("\n" + "=" * 50)
            print("All tests passed! ✅")
            return 0
        else:
            print("\n" + "=" * 50)
            print("Some tests failed! ❌")
            return 1

    except Exception as e:
        print(f"\nError running tests: {e}")
        return 1
    finally:
        loop.close()

if __name__ == "__main__":
    exit(main())
