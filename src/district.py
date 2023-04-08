from agent import Agent
import numpy as np
from collections import Counter

class District:
    """
    An object representing a district that holds an election
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

    def nominate(self, parties=None):
        if (parties != None):
            party_opis = [party.x for party in parties]
            party_sd = parties[0].sd

            tmp_opis = np.array([resident.x for resident in self.residents])
            diff = np.abs(np.subtract.outer(party_opis, tmp_opis))
            diff_square = np.square(diff)
            gaussian_filter = np.exp((-diff_square)/(2*np.square(party_sd)))
            party_nom = np.random.binomial(n=1, p=gaussian_filter).astype(bool) # party selection of residents
            self_nom = np.random.binomial(n=1, p=self.nom_rate, size=self.N).astype(bool) # resident self-nomination

            party_msks = (party_nom & self_nom)

            # assigning candidates w/ more than 1 parties to a specific party (not necessary yet)
            multip_msks = party_msks.sum(axis=0) > 1
            multip_ids = np.arange(self.N)[multip_msks]

            party_ids = np.arange(len(parties))

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
        non_candidate_opis = np.array([resident.x for resident in self.residents[~self.nom_msks]])
        vote = []

        # Deterministic voting
        # reimplementing with numpy operations
        if voting == "deterministic":
            vote = np.argmin(np.abs(np.subtract.outer(non_candidate_opis, candidate_opis)), axis=1)

        # Probabilistic voting
        else:
            opi_diffs = np.abs(np.subtract.outer(non_candidate_opis, candidate_opis))
            elect_probs = 1 - (opi_diffs / opi_diffs.sum(axis=1)[:,None])
            elect_cumprobs = elect_probs.cumsum(axis=1)
            tmp_randnums = np.random.uniform(0, 1, size=non_candidate_opis.shape[0])

            vote = (elect_cumprobs < tmp_randnums[:,None]).sum(axis=1)

        vote_counter = Counter(vote)
        self.elected.extend([candidates[id] for (id, vote_count) in vote_counter.most_common(self.rep_num)])