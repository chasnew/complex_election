from agent import Agent
import numpy as np
from collections import Counter

class Election:
    """
    Represents an electoral system
    """

    def __init__(self, N, nom_rate = 0.05, rep_num = 1,
                 party_num = None, party_sd = 0.2,
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
        self.party_num = party_num

        if (party_num != None) and (party_num > 0):
            self.parties = np.random.uniform(-1, 1, size=self.party_num)
            self.party_sd = party_sd

        self.nom_msks = []
        self.elected = []

    def nominate(self):
        if (self.party_num != None) and (self.party_num > 0):
            tmp_opis = np.array([resident.x for resident in self.residents])
            diff = np.abs(np.subtract.outer(self.parties, tmp_opis))
            diff_square = np.square(diff)
            gaussian_filter = np.exp((-diff_square)/(2*np.square(self.party_sd)))
            party_nom = np.random.binomial(n=1, p=gaussian_filter).astype(bool) # party selection of residents
            self_nom = np.random.binomial(n=1, p=self.nom_rate, size=self.N).astype(bool) # resident self-nomination

            party_msks = (party_nom & self_nom)

            # assigning candidates w/ more than 1 parties to a specific party (not necessary yet)
            multip_msks = party_msks.sum(axis=0) > 1
            multip_ids = np.arange(self.N)[multip_msks]

            party_ids = np.arange(self.party_num)

            for i in range(multip_ids.shape[0]):
                indv_diff = diff[party_msks[:,i],i]
                party_probs = (indv_diff / np.sum(indv_diff))
                party = np.random.choice(party_ids[party_msks[:,i]],
                                         size = 1, p = party_probs)

                party_msks[:,i] = False
                party_msks[party,i] = True

            self.nom_msks = np.any(party_msks, axis=0)
        else:
            self.nom_msks = np.random.binomial(n=1, p=self.nom_rate, size=self.N).astype(bool)

    def vote(self, voting="deterministic"):

        candidates = self.residents[self.nom_msks]
        candidate_opis = np.array([candidate.x for candidate in candidates])
        vote = []

        # Deterministic voting
        if voting == "deterministic":
            for resident in self.residents[~self.nom_msks]:
                vote.append(np.argmin(np.abs(candidate_opis - resident.x)))

        # Probabilistic voting
        else:
            for resident in self.residents[~self.nom_msks]:
                opi_diffs = np.abs(candidate_opis - resident.x)
                vote_probs = opi_diffs / np.sum(opi_diffs)
                vote.append(np.random.choice(np.arange(len(candidates))), size=1, p=vote_probs)

        vote_counter = Counter(vote)
        self.elected.extend([candidates[id] for (id, vote_count) in vote_counter.most_common(self.rep_num)])

    def step(self, voting="deterministic"):
        self.nominate()
        self.vote(voting=voting)

    def __repr__(self):
        """
        Text representation of the model.
        """
        return f'Population (size: {self.N}|party: {self.party_num})'
