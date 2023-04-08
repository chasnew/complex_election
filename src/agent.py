class Agent:
    """
    Represents a single person with an ID and the person's opinon or policy position.
    """

    def __init__(self, id_: str, x):
        """
        Initializes the agent.

        Parameters
        ----------
        id_ : str
            The ID of the agent.
        """
        self.id = id_
        self.x = x

    def __repr__(self):
        """
        Text representation of the agent.
        """
        return f'Agent ({self.id}|{self.x})'
