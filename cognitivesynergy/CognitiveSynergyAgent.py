import redis
import json
import threading
import time
from ONA.DockerInteractor import DockerInteractor
from utils.DVDisplayChannel import DVDisplayChannel, DVMessage
import random
import timeit

class CognitiveSynergyAgent:
    def __init__(self, agent_type, config):
        self.agent_type = agent_type
        self.config = config
        self.is_running = False
        self._thread = None
        self.ona = DockerInteractor(['docker', 'exec', '-i', '-w', '/app/misc/Python', 'ONA', 'python3', 'NAR_json.py'])
        self.ona_terminator = "\"requestOutputArgs\": false}"
        self.display_channel = DVDisplayChannel("ONA_DISPLAY")

    def execute_command(self, command):
        response = self.ona.execute_command(command, self.ona_terminator)
        return response

    def test_ona_performance(self):
        start_time = timeit.default_timer()
        for _ in range(10000):
            response = self.execute_command("<cat --> furry_animal>.\n<cat --> furry_animal>?")
        end_time = timeit.default_timer()
        time_taken = end_time - start_time  # time_taken is in seconds
        messages_per_second = 10000.0 / time_taken
        print(f"Messages per second: {messages_per_second}")
        print(f"sample message: {response}")

    def _run(self):
        iterations = 0
        iterations2 = 0
        print(f"Agent of type {self.agent_type} is starting...")

        while self.is_running:
            # Process data and publish results to Redis
            data = {}  # Your data here
            #self.redis_client.publish('channel_name', json.dumps(data))
            iterations = iterations + 1
            iterations2 = iterations2 + 1
            msgs = []
            msgs.append(DVMessage(f"Agent of type {self.agent_type} is running...iteration {iterations}", 
                            text_position={'x': 10, 'y': 50}, 
                            bounding_box={'x1': 0, 'y1': 0, 'x2': 100, 'y2': 100}, 
                            color='red', 
                            line_width=2, 
                            font_size=20, 
                            font_color='rgb(0,0,255)'))
            if iterations2 < 50:
                msgs.append(DVMessage(f"Analysing the scene...",
                            text_position={'x': 10, 'y': 100},
                            font_color='rgb(0,255,0)'))
            elif iterations2 < 100:
                msgs.append(DVMessage(f"Done analysing the scene.",
                            text_position={'x': 10, 'y': 100},
                            font_color='rgb(0,255,0)'))
            else:
                iterations2 = 0

            self.display_channel.write_message(msgs)
            response = self.execute_command("<cat --> furry_animal>.\n<cat --> furry_animal>?")
            print(f"Response: {response}")
            time.sleep(.5)

        print(f"Agent of type {self.agent_type} has stopped.")

    def start(self):
        # Start the agent
        self.is_running = True
        self._thread = threading.Thread(target=self._run)
        self._thread.start()

    def stop(self):
        # Stop the agent
        self.is_running = False
        if self._thread is not None:
            self._thread.join()
