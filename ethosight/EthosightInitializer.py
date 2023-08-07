import threading
from Ethosight import Ethosight


class EthosightSingleton:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = Ethosight(reasoner='reasoner')
        return cls._instance
