class CognitiveSynergyAgentManager:
    def __init__(self):
        # Initialize an empty list or dict to store the agents
        self.agents = []

    def createAgent(self, agent_type, config):
        # Create an agent of the specified type with the given config
        agent = CognitiveSynergyAgent(config)
        self.agents.append(agent)

    def startAgent(self, agent_index):
        # Start the specified agent
        self.agents[agent_index].start()

    def stopAgent(self, agent_index):
        # Stop the specified agent
        self.agents[agent_index].stop()

