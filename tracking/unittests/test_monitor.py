import time
import unittest

from unittest import mock
import sys
from unittest.mock import patch

from tracking.Monitor import GPUCalculator

redis_mock = mock.MagicMock()
sys.modules['redis'] = redis_mock
# import Monitor as monitor


class TestMonitor(unittest.TestCase):
    """ The function to test (would usually be loaded from a module outside this file).
    """

    def test_MMTMonitor_starttimer(self):
        redis_mock = mock.MagicMock()
        import tracking.Monitor as monitor
        obj = monitor.MMTMonitor(redis_mock, 'key')
        obj.start_timer()
        self.assertIsNotNone(obj.start_time)

    def test_MMTMonitor_with_latency(self):
        redis_mock = mock.MagicMock()
        import tracking.Monitor as monitor
        obj = monitor.MMTMonitor(redis_mock, 'key')
        obj.start_timer()
        time.sleep(1)
        obj.end_timer()
        print(obj.average_latency)
        self.assertNotEqual(obj.average_latency, 0.0)

    def test_MMTMonitor_counter_15(self):
        redis_mock = mock.MagicMock()
        import tracking.Monitor as monitor
        obj = monitor.MMTMonitor(redis_mock, 'key')
        obj.start_timer()
        obj.counter = 14
        obj.end_timer()
        print(obj.average_latency)
        self.assertEqual(obj.average_latency, 0)

    def test_GPUCalculator(self):
        gpu_calculator = GPUCalculator(redis_mock)
        gpu_calculator.add()
        self.assertEqual(gpu_calculator.count, 1)

    @patch('tracking.Monitor.subprocess')
    def test_GPUCalculator_with_count_15(self, mock_sub):
        output = '1, 2, 3, 4, 5'.encode('utf-8')
        mock_sub.check_output.return_value = output
        gpu_calculator = GPUCalculator(redis_mock)
        gpu_calculator.count = 14
        gpu_calculator.add()
        self.assertEqual(gpu_calculator.count, 15)
