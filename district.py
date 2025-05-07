from agent import Resident, Candidate
import numpy as np
import scipy.stats as stats
from collections import Counter

class District:
    """
    An object representing a district that holds an election
    """

    def __init__(self, d_id, N, party_num = None,
                 nom_rate = 5, rep_num = 1,
                 alpha = 0.5, beta = 0.5,
                 opinion_dist_dict = {'dist': "uniform", 'low': -1, 'up': 1}):
        """
        Initializes the election model.

        Parameters
        ----------
        d_id: district id
        N : number of residents (agents)
        party_num: number of political parties
        nom_rate: fixed number of nominated candidates (per party if parties exist)
        rep_num: number of representatives
        alpha: memory bias when evaluating winning probability of each party
        opinion_distribution: distribution of opinions of the district residents
        gaussian_mu: mean of the gaussian distribution of opinions
        gaussian_sd: standard deviation of the gaussian distribution of opinions
        """

        self.d_id = d_id

        if opinion_dist_dict['dist'] == "uniform":
            init_opis = np.random.uniform(opinion_dist_dict['low'],
                                          opinion_dist_dict['up'], size=N)
        elif opinion_dist_dict['dist'] == "gaussian":
            lower, upper = -1, 1
            gaussian_mu = opinion_dist_dict['mu']
            gaussian_sd = opinion_dist_dict['sd']
            a = (lower - gaussian_mu) / gaussian_sd  # lower sd cutoff
            b = (upper - gaussian_mu) / gaussian_sd  # upper sd cutoff
            init_opis = stats.truncnorm(a, b, loc=gaussian_mu, scale=gaussian_sd).rvs(N)

        self.residents = np.array([Resident(id_=i, d_id=self.d_id, x=init_opis[i]) for i in range(N)])
        self.N = N
        self.nom_rate = nom_rate
        self.rep_num = rep_num
        self.alpha = alpha
        self.beta = beta

        self.loc_candidates = []
        self.elected = []
        self.elected_party = []
        self.cum_elected = []
        self.cum_elected_party = []

        self.vote_masks = []

        # vote proportion of each party in previous election cycle (initially 0 for all)
        self.prev_vote_props = np.zeros(party_num)


    def nominate(self, parties=[]):
        if (len(parties) > 0):
            party_opis = [party.x for party in parties]
            party_sd = parties[0].sd


            tmp_opis = np.array([resident.x for resident in self.residents])
            diff = np.abs(np.subtract.outer(party_opis, tmp_opis)) # differences in preferences

            # gaussian filter of party selection
            diff_square = np.square(diff)
            gaussian_filter = np.exp((-diff_square)/(2*np.square(party_sd)))

            self.nom_msks = np.zeros(self.N).astype(bool)

            # party member list is extended with local candidates
            for party in parties:

                pcandidate_pos = np.random.normal(loc=party.x, scale=party_sd, size=self.nom_rate)
                district_candidates = [Candidate(i, self.d_id, candidate_pos, party.id)
                                       for i, candidate_pos in enumerate(pcandidate_pos)]
                party.members.extend(district_candidates)

                self.loc_candidates.extend(district_candidates)
                # self.nom_msks[nom_party_inds] = True

        else:
            nom_inds = np.random.choice(np.arange(self.N), size=self.nom_rate, replace=False)
            district_candidates = [Candidate(resident.id, self.d_id, resident.x, None)
                                   for resident in self.residents[nom_inds]]
            self.loc_candidates.extend(district_candidates)


    def vote(self, voting="deterministic", parties=[],
             strategic=False, party_filter=False):

        candidates = self.loc_candidates # self.residents[self.nom_msks]
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
                pd_candidates = np.array([candidate for candidate in self.loc_candidates
                                          if candidate.party_id == pid])
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

            # Strategic voting
            if len(parties) > 2 and strategic:
                vote = self.strategic_vote(resident_opis, candidates, parties)

            # Naive voting
            else:
                # residents vote for the closest candidate
                vote = np.argmin(np.abs(np.subtract.outer(resident_opis, candidate_opis)), axis=1)

            vote_counter = Counter(vote)
            self.prev_vote_props = np.array([vote_counter[i] / vote.shape[0] for i in range(len(parties))])
            # print('district {}, vote prop', self.prev_vote_props)

            winner_id = vote_counter.most_common(1)[0][0]

            # mark winning candidate as being elected
            candidates[winner_id].elected = True

            self.elected = [candidates[winner_id]]
            self.cum_elected.extend(self.elected)

            self.elected_party = [winner_id]
            self.cum_elected_party.extend(self.elected_party)

        # Proportional representation; open list (proportional_rep)
        # All candidates are on the ballots but share winnability of the party
        else:
            # all local party candidates (iterate over each party)
            district_candidates = self.loc_candidates

            # Strategic voting
            if len(parties) > 2 and strategic:
                # may need an argument for proportional_rep since number of parties doesn't match candidates
                vote = self.strategic_vote(resident_opis, district_candidates, parties)

            # Naive voting
            else:
                # residents vote for the closest candidate
                candidate_opis = [candidate.x for candidate in district_candidates]
                vote = np.argmin(np.abs(np.subtract.outer(resident_opis, candidate_opis)), axis=1)

                # map candidate id to party id
                cand_party_map = {i: [] for i in range(len(parties))}
                for cid, candidate in enumerate(district_candidates):
                    c_party = candidate.party_id
                    cand_party_map[c_party].append(cid)

                for i in range(len(parties)):
                    vote[np.isin(vote, cand_party_map[i])] = i

            vote_counter = Counter(vote)
            self.prev_vote_props = np.array([vote_counter[i] / vote.shape[0] for i in range(len(parties))])

            # Calculate seats for each party (Hamilton's method)
            # Previously d'Hondt method
            party_rep_nums = {}
            remainders = [0 for _ in range(len(vote_counter))]

            total_vote = sum(vote_counter.values())
            hare_quota = total_vote / self.rep_num

            for pid, vote_count in vote_counter.items():
                tmp = vote_count / hare_quota
                party_rep_nums[pid] = int(tmp)
                remainders[pid] = tmp - int(tmp)

            remain_seats = self.rep_num - sum(party_rep_nums.values())
            top_remain_parties = np.argsort(remainders)

            # print('Allocated seat num w/ full quota: ', sum(party_rep_nums.values()))

            for i in range(remain_seats):
                party_rep_nums[top_remain_parties[i]] += 1

            # print('Seats allocation: {}'.format(party_rep_nums))

            self.elected = []
            self.elected_party = []

            # Parties select their candidates from their party pool to occupy the seats
            for (pid, elected_num) in party_rep_nums.items():

                # local party candidates
                p_candidates = np.array([candidate for candidate in parties[pid].members
                                         if candidate.d_id == self.d_id])

                if party_filter:
                    candidate_opis = np.array([candidate.x for candidate in p_candidates])

                    diff_square = np.square(np.abs(candidate_opis - parties[pid].x))
                    gaussian_filter = np.exp((-diff_square) / (2 * np.square(parties[pid].sd)))
                    elect_probs = gaussian_filter / np.sum(gaussian_filter)
                else:
                    elect_probs = np.ones(len(p_candidates)) / len(p_candidates)

                # if number of seats is greater than the party size
                if elected_num > len(p_candidates):
                    elect_ids = np.arange(len(p_candidates))
                else:
                    elect_ids = np.random.choice(np.arange(len(p_candidates)), p=elect_probs,
                                                 replace=False, size=elected_num)

                # mark candidate as being elected
                for elected_c in p_candidates[elect_ids]:
                    elected_c.elected = True

                self.elected.extend(list(p_candidates[elect_ids]))
                self.elected_party.extend([pid] * elected_num)

            self.cum_elected.extend(self.elected)
            self.cum_elected_party.extend(self.elected_party)


    '''
    Strategic voting results based on previous election results as well as polling
    '''
    def strategic_vote(self, resident_opis, candidates, parties):

        candidate_opis = []
        candidate_party = []
        for candidate in candidates:
            candidate_opis.append(candidate.x)
            candidate_party.append(candidate.party_id)

        candidate_opis = np.array(candidate_opis)
        candidate_party = np.array(candidate_party)

        # preference alignment
        dists = np.abs(np.subtract.outer(resident_opis, candidate_opis))

        # clip off parties with farthest distance
        # max_dists = np.max(dists, axis=1)
        # max_masks = dists.T == max_dists
        # dists.T[max_masks] = 2

        # reverse and re-scale distance from (0,2) -> (1,0)
        rescaled_dists = 1 - (dists / 2)

        # winnability calculation
        party_num = len(parties)

        # ideal polling
        sincere_vote = np.argmin(dists, axis=1)
        sincere_vote = candidate_party[sincere_vote] # map candidate to their party
        sincere_vote_count = Counter(sincere_vote)

        poll_props = np.array([sincere_vote_count[i] / sincere_vote.shape[0]
                               for i in range(party_num)])
        # print('poll proportion: ', poll_props)
        # print('history record: ', self.prev_vote_props)

        # propensity to not waste vote
        weighted_vote_props = (self.alpha * self.prev_vote_props) + ((1 - self.alpha) * poll_props)
        # print('district {}, weighted vote props:'.format(self.d_id), weighted_vote_props)

        # (1 / (number of seats + 1)) droop quota -> winning probability
        crit_thresh = 1 / (self.rep_num + 1)
        win_probs = np.clip((1 / crit_thresh) * weighted_vote_props, a_min=None, a_max=1)

        # final vote decision
        # change from product to weighted sum rescaled_dists * win_probs

        # proportional representation has more candidates than parties
        if len(candidates) > party_num:
            win_probs = win_probs[candidate_party]

        vote = np.argmax(((1 - self.beta) * rescaled_dists) + (self.beta * win_probs), axis=1)

        if len(candidates) > party_num:
            vote = candidate_party[vote] # map candidate to their party

        return vote