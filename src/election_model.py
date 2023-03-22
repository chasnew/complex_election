from agent import Agent
import numpy as np
from collections import Counter

class Election:
    """
    Represents an electoral system
    """

    def __init__(self, N, nom_rate = 0.05, rep_num = 1,
                 opinion_distribution = "uniform"):
        """
        Initializes the election model.

        Parameters
        ----------
        N : number of residents (agents)
        nom_rate: nomination rate (the rate at which residents become political candidates)
        rep_num: number of representatives
        """

        init_opis = np.random.uniform(-1, 1, size=N)
        self.residents = np.array([Agent(i, init_opis[i]) for i in range(N)])
        self.N = N
        self.nom_rate = nom_rate
        self.rep_num = rep_num

        self.nom_msks = []
        self.elected = []

    def nominate(self):
        self.nom_msks = np.random.choice([True, False], size=self.N, p=[self.nom_rate, 1-self.nom_rate])

    def vote(self):
        # Deterministic voting
        candidates = self.residents[self.nom_msks]
        candidate_opis = np.array([candidate.x for candidate in candidates])
        vote = []
        for resident in self.residents[~self.nom_msks]:
            vote.append(np.argmin(np.abs(candidate_opis - resident.x)))

        vote_counter = Counter(vote)
        self.elected.extend([candidates[id] for (id, vote_count) in vote_counter.most_common(self.rep_num)])

    def step(self):
        self.nominate()
        self.vote()

    def __repr__(self):
        """
        Text representation of the model.
        """
        return f'Population (size: {self.N})'
