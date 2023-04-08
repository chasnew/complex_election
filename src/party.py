class Party:
    """
    Represents a party with an ID and party's position and its inclusiveness.
    """

    def __init__(self, id_: str, x, inclusiveness):
        """
        Initializes the agent.

        Parameters
        ----------
        id_ : str
            The ID of the agent.
        """
        self.id = id_
        self.x = x
        self.sd = inclusiveness

    def __repr__(self):
        """
        Text representation of the agent.
        """
        return f'Party ({self.id}|{self.x}|{self.sd})'
