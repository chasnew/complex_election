class Resident:
    """
    Represents a resident with an ID and the person's opinion or policy position.
    """

    def __init__(self, id_, d_id, x, trust=1):
        """
        Initializes the agent.

        Parameters
        ----------
        id_: The ID of the agent.
        d_id: district ID of the agent
        x: opinion value or policy preference of the agent
        trust: electoral trust that affects political participation
        """
        self.id = id_
        self.d_id = d_id
        self.x = x
        self.trust = trust

    def __repr__(self):
        """
        Text representation of the agent.
        """
        return f'Agent ({self.id}|{self.d_id}|{self.x})'



class Candidate(Resident):
    """
    Represents a nominated political candidate with an assigned party and status of whether they got elected
    """

    def __init__(self, id_, d_id, x, party_id):
        """
        Initializes the agent.

        Parameters
        ----------
        id_: The ID of the agent.
        d_id: district ID of the agent
        x: opinion value or policy preference of the agent
        """
        super().__init__(id_, d_id, x)
        self.party_id = party_id
        self.elected = False

    def __repr__(self):
        """
        Text representation of the agent.
        """
        return f'Candidate ({self.id}|{self.d_id}|{self.x}|{self.party_id})'
