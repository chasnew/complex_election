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
        opinion_distribution: distribution of opinions of the district residents
        """

        init_opis = np.random.uniform(-1, 1, size=N)
        self.residents = np.array([Agent(i, init_opis[i]) for i in range(N)])
        self.N = N
        self.nom_rate = nom_rate
        self.rep_num = rep_num

        self.nom_msks = []
        self.elected = []
        self.elected_party = []

    def nominate(self, parties=[]):
        if (len(parties) > 0):
            party_opis = [party.x for party in parties]
            party_sd = parties[0].sd

            tmp_opis = np.array([resident.x for resident in self.residents])
            diff = np.abs(np.subtract.outer(party_opis, tmp_opis))
            diff_square = np.square(diff)
            gaussian_filter = np.exp((-diff_square)/(2*np.square(party_sd)))
            party_nom = np.random.binomial(n=1, p=gaussian_filter).astype(bool) # party selection of residents
            self_nom = np.random.binomial(n=1, p=self.nom_rate, size=self.N).astype(bool) # resident self-nomination

            party_msks = (party_nom & self_nom)

            # assigning candidates w/ more than 1 parties to a specific party
            multip_msks = party_msks.sum(axis=0) > 1
            multip_ids = np.arange(self.N)[multip_msks]

            party_ids = np.arange(len(parties))

            for i in multip_ids:
                # extract diff in opinions of candidate and parties that recruit them
                indv_diff = diff[party_msks[:,i],i]
                party_probs = (indv_diff / np.sum(indv_diff))

                # print('party_probability = {}'.format(party_probs))
                # print(party_msks[:,i])

                party = np.random.choice(party_ids[party_msks[:,i]],
                                         size = 1, p = party_probs)

                party_msks[:,i] = False
                party_msks[party,i] = True

            self.nom_msks = np.any(party_msks, axis=0)

            for party in parties:
                party.members = self.residents[party_msks[party.id,:]]
        else:
            self.nom_msks = np.random.binomial(n=1, p=self.nom_rate, size=self.N).astype(bool)

    def vote(self, voting="deterministic", parties=[], party_filter=False):

        candidates = self.residents[self.nom_msks]
        candidate_opis = np.array([candidate.x for candidate in candidates])
        non_candidate_opis = np.array([resident.x for resident in self.residents[~self.nom_msks]])
        vote = []

        # Deterministic voting
        if voting == "deterministic":
            vote = np.argmin(np.abs(np.subtract.outer(non_candidate_opis, candidate_opis)), axis=1)

            vote_counter = Counter(vote)
            self.elected.extend([candidates[id] for (id, vote_count) in vote_counter.most_common(self.rep_num)])

        # Probabilistic voting
        elif voting == "probabilistic":
            opi_diffs = np.abs(np.subtract.outer(non_candidate_opis, candidate_opis))
            elect_probs = 1 - (opi_diffs / opi_diffs.sum(axis=1)[:,None])
            elect_cumprobs = elect_probs.cumsum(axis=1)
            tmp_randnums = np.random.uniform(0, 1, size=non_candidate_opis.shape[0])

            vote = (elect_cumprobs < tmp_randnums[:,None]).sum(axis=1)

            vote_counter = Counter(vote)
            self.elected.extend([candidates[id] for (id, vote_count) in vote_counter.most_common(self.rep_num)])

        # Single candidate per party (First-past-the-post)
        elif voting == "one_per_party":
            candidates = np.full(len(parties), 0)
            candidate_opis = np.full(len(parties), 0)

            # Parties nominate one candidate to enter the race
            for i in len(parties):
                id = parties[i].id

                if party_filter:
                    party_candidate_opis = np.array([candidate.x for candidate in parties[id].members])

                    diff_square = np.square(np.abs(party_candidate_opis - parties[id].x))
                    gaussian_filter = np.exp((-diff_square) / (2 * np.square(parties[id].sd)))
                    party_nom_probs = gaussian_filter / np.sum(gaussian_filter)
                else:
                    party_nom_probs = np.ones(len(parties[id].members)) / len(parties[id].members)

                candidates[id] = np.random.choice(parties[id].members, p=party_nom_probs, size=1)
                candidate_opis[id] = candidates[id].x

            # party members / politicians do not vote
            # residents vote for the closest candidate
            vote = np.argmin(np.abs(np.subtract.outer(non_candidate_opis, candidate_opis)), axis=1)

            vote_counter = Counter(vote)
            self.elected.extend([candidates[id] for (id, vote_count) in vote_counter.most_common(self.rep_num)])
            self.elected_party.extend([id for (id, vote_count) in vote_counter.most_common(self.rep_num)])

        # Proportional representation (closed list)
        else:
            party_opis = [party.x for party in parties]

            # residents vote for the closest party
            vote = np.argmin(np.abs(np.subtract.outer(non_candidate_opis, party_opis)), axis=1)

            vote_counter = Counter(vote)
            party_rep_nums = {id: int(np.round((vote_count/self.N)*self.rep_num))
                              for (id, vote_count) in vote_counter.items()}

            # Parties select their candidates to occupy the seats
            for (id, elected_num) in party_rep_nums.items():

                if party_filter:
                    candidate_opis = np.array([candidate.x for candidate in parties[id].members])

                    diff_square = np.square(np.abs(candidate_opis - parties[id].x))
                    gaussian_filter = np.exp((-diff_square) / (2 * np.square(parties[id].sd)))
                    elect_probs = gaussian_filter / np.sum(gaussian_filter)
                else:
                    elect_probs = np.ones(len(parties[id].members)) / len(parties[id].members)

                elect_ids = np.random.choice(np.arange(candidate_opis.shape[0]), p=elect_probs, size=elected_num)

                self.elected.extend(list(parties[id].members[elect_ids]))
                self.elected_party.extend([id] * elected_num)