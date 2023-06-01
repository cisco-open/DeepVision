from CognitiveSynergyAgent import CognitiveSynergyAgent

class CognitiveSynergyAgentManager:
    def __init__(self):
        # Initialize an empty list or dict to store the agents
        self.agents = []

    def createAgent(self, agent_type, config):
        # Create an agent of the specified type with the given config
        agent = CognitiveSynergyAgent(agent_type="Basic",config={})
        self.agents.append(agent)
        return agent
