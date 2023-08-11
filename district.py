from agent import Resident, Candidate
import numpy as np
import scipy.stats as stats
from collections import Counter

class District:
    """
    An object representing a district that holds an election
    """

    def __init__(self, d_id, N, nom_rate = 0.05, rep_num = 1,
                 opinion_distribution = "uniform",
                 gaussian_mu = 0, gaussian_sd = 0.5):
        """
        Initializes the election model.

        Parameters
        ----------
        d_id: district id
        N : number of residents (agents)
        nom_rate: nomination rate (the rate at which residents become political candidates)
        rep_num: number of representatives
        opinion_distribution: distribution of opinions of the district residents
        gaussian_mu: mean of the gaussian distribution of opinions
        gaussian_sd: standard deviation of the gaussian distribution of opinions
        """

        self.d_id = d_id

        if opinion_distribution == "uniform":
            init_opis = np.random.uniform(-1, 1, size=N)
        elif opinion_distribution == "gaussian":
            lower, upper = -1, 1
            a = (lower - gaussian_mu) / gaussian_sd  # lower sd cutoff
            b = (upper - gaussian_mu) / gaussian_sd  # upper sd cutoff
            init_opis = stats.truncnorm(a, b, loc=gaussian_mu, scale=gaussian_sd).rvs(N)

        self.residents = np.array([Resident(i, self.d_id, init_opis[i]) for i in range(N)])
        self.N = N
        self.nom_rate = nom_rate
        self.rep_num = rep_num

        self.nom_msks = []
        self.elected = []
        self.elected_party = []
        self.cum_elected = []
        self.cum_elected_party = []


    def nominate(self, parties=[]):
        if (len(parties) > 0):
            party_opis = [party.x for party in parties]
            party_sd = parties[0].sd

            tmp_opis = np.array([resident.x for resident in self.residents])
            diff = np.abs(np.subtract.outer(party_opis, tmp_opis))

            # pre-select residents to be core members of the party ensuring that parties have at least one candidate
            top_candidates = np.argpartition(diff, len(parties))[:,:len(parties)]
            core_list = []

            for i in range(len(parties)):
                for j in range(len(top_candidates)):
                    if top_candidates[i,j] not in core_list:
                        core_list.append(top_candidates[i,j])
                        break

            # gaussian filter of party selection
            diff_square = np.square(diff)
            gaussian_filter = np.exp((-diff_square)/(2*np.square(party_sd)))
            party_nom = np.random.binomial(n=1, p=gaussian_filter).astype(bool) # party selection of residents
            self_nom = np.random.binomial(n=1, p=self.nom_rate, size=self.N).astype(bool) # resident self-nomination

            party_msks = (party_nom & self_nom)

            # setting mask for core members of the parties
            for i in range(len(parties)):
                party_msks[:,core_list[i]] = False
                party_msks[i,core_list[i]] = True

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

            # party member list is extended with local candidates
            for party in parties:
                district_candidates = [Candidate(resident.id, self.d_id, resident.x, party.id)
                                       for resident in self.residents[party_msks[party.id,:]]]
                party.members.extend(district_candidates)
        else:
            self.nom_msks = np.random.binomial(n=1, p=self.nom_rate, size=self.N).astype(bool)


    def vote(self, voting="deterministic", parties=[], party_filter=False):

        candidates = self.residents[self.nom_msks]
        candidate_opis = np.array([candidate.x for candidate in candidates])

        # every resident votes
        resident_opis = np.array([resident.x for resident in self.residents])

        # Deterministic voting
        if voting == "deterministic":
            vote = np.argmin(np.abs(np.subtract.outer(resident_opis, candidate_opis)), axis=1)

            vote_counter = Counter(vote)
            self.elected = [candidates[id] for (id, vote_count) in vote_counter.most_common(self.rep_num)]
            self.cum_elected.extend(self.elected)

        # Probabilistic voting
        elif voting == "probabilistic":
            opi_diffs = np.abs(np.subtract.outer(resident_opis, candidate_opis))
            elect_probs = 1 - (opi_diffs / opi_diffs.sum(axis=1)[:,None])
            elect_cumprobs = elect_probs.cumsum(axis=1)
            tmp_randnums = np.random.uniform(0, 1, size=resident_opis.shape[0])

            vote = (elect_cumprobs < tmp_randnums[:,None]).sum(axis=1)

            vote_counter = Counter(vote)
            self.elected = [candidates[id] for (id, vote_count) in vote_counter.most_common(self.rep_num)]
            self.cum_elected.extend(self.elected)

        # Single candidate per party (First-past-the-post)
        elif voting == "one_per_party":
            candidates = [0] * len(parties)
            candidate_opis = np.full(len(parties), 0).astype(np.float64)

            # Parties nominate one candidate to enter the race
            for i in range(len(parties)):
                pid = parties[i].id

                # local party candidates in the district
                pd_candidates = np.array([candidate for candidate in parties[pid].members
                                                 if candidate.d_id == self.d_id])
                if party_filter:
                    party_candidate_opis = np.array([candidate.x for candidate in pd_candidates])

                    diff_square = np.square(np.abs(party_candidate_opis - parties[pid].x))
                    gaussian_filter = np.exp((-diff_square) / (2 * np.square(parties[pid].sd)))
                    party_nom_probs = gaussian_filter / np.sum(gaussian_filter)
                else:
                    party_nom_probs = np.ones(len(pd_candidates)) / len(pd_candidates)

                candidates[pid] = np.random.choice(pd_candidates, p=party_nom_probs, size=1)[0]
                candidate_opis[pid] = candidates[pid].x

            candidates = np.array(candidates)

            # residents vote for the closest candidate
            vote = np.argmin(np.abs(np.subtract.outer(resident_opis, candidate_opis)), axis=1)

            vote_counter = Counter(vote)
            winner_id = vote_counter.most_common(1)[0][0]

            # mark winning candidate as being elected
            candidates[winner_id].elected = True

            self.elected = candidates[winner_id]
            self.cum_elected.append(self.elected)

            self.elected_party = winner_id
            self.cum_elected_party.append(self.elected_party)

        # Proportional representation (closed list)
        else:
            party_opis = [party.x for party in parties]

            # residents vote for the closest party
            vote = np.argmin(np.abs(np.subtract.outer(resident_opis, party_opis)), axis=1)

            vote_counter = Counter(vote)
            party_rep_nums = {id: int(np.round((vote_count/self.N)*self.rep_num))
                              for (id, vote_count) in vote_counter.items()}

            # Parties select their candidates from their party pool to occupy the seats
            for (pid, elected_num) in party_rep_nums.items():

                # party candidates who haven't been elected yet
                p_candidates = np.array([candidate for candidate in parties[pid].members
                                         if candidate.elected == False])

                if party_filter:
                    candidate_opis = np.array([candidate.x for candidate in p_candidates])

                    diff_square = np.square(np.abs(candidate_opis - parties[pid].x))
                    gaussian_filter = np.exp((-diff_square) / (2 * np.square(parties[pid].sd)))
                    elect_probs = gaussian_filter / np.sum(gaussian_filter)
                else:
                    elect_probs = np.ones(len(p_candidates)) / len(p_candidates)

                elect_ids = np.random.choice(np.arange(len(p_candidates)), p=elect_probs, size=elected_num)

                # mark candidate as being elected
                for elected_c in p_candidates[elect_ids]:
                    elected_c.elected = True

                self.elected = list(p_candidates[elect_ids])
                self.cum_elected.extend(self.elected)

                self.elected_party = [pid] * elected_num
                self.cum_elected_party.extend(self.elected_party)

    def appraise(self, elected_pool):

        # Average position of the representatives
        elected_position = np.mean([elected.x for elected in elected_pool])

        resident_opis = np.array([resident.x for resident in self.residents])
        opi_diffs = np.abs(resident_opis - elected_position)

        # k = steepness of logit
        # re_loc is re-centering the logit function
        k = -5
        re_loc = 1
        logit = 1 / (1 + np.exp(-k*(opi_diffs - re_loc)))
        trust_update = 0.5 - logit # flip logit horizontally and readjust min, max (-0.5, 0.5)

        for resident in self.residents:
            resident.trust = resident.trust + trust_update

        # what if no agents vote?

