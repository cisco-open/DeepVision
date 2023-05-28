from CognitiveSynergyAgent import CognitiveSynergyAgent
from CognitiveSynergyAgentManager import CognitiveSynergyAgentManager

def main():
    # Create an instance of the CognitiveSynergyAgentManager
    agent_manager = CognitiveSynergyAgentManager()


    # Start the agent
    agent = agent_manager.createAgent(agent_type="Basic", config={})
    agent.start()

if __name__ == "__main__":
    main()
