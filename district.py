from agent import Resident, Candidate
import numpy as np
import scipy.stats as stats
from collections import Counter

class District:
    """
    An object representing a district that holds an election
    """

    def __init__(self, d_id, N, nom_rate = 5, rep_num = 1, alpha = 0.5,
                 opinion_distribution = "uniform",
                 gaussian_mu = 0, gaussian_sd = 0.5):
        """
        Initializes the election model.

        Parameters
        ----------
        d_id: district id
        N : number of residents (agents)
        nom_rate: fixed number of nominated candidates (per party if parties exist)
        rep_num: number of representatives
        alpha: memory bias when evaluating winning probability of each party
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

        self.residents = np.array([Resident(id_=i, d_id=self.d_id, x=init_opis[i]) for i in range(N)])
        self.N = N
        self.nom_rate = nom_rate
        self.rep_num = rep_num
        self.alpha = alpha

        self.nom_msks = []
        self.elected = []
        self.elected_party = []
        self.cum_elected = []
        self.cum_elected_party = []

        self.vote_masks = []
        self.prev_vote_props = [] # vote proportion of each party in previous election cycle


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
                avail_inds = np.arange(self.N)[~self.nom_msks] # filter out already recruited residents

                avail_gaussian = gaussian_filter[party.id][avail_inds]
                nom_party_probs = avail_gaussian / np.sum(avail_gaussian)
                nom_party_inds = np.random.choice(avail_inds, size=self.nom_rate,
                                                  replace=False, p=nom_party_probs)
                district_candidates = [Candidate(resident.id, self.d_id, resident.x, party.id)
                                       for resident in self.residents[nom_party_inds]]
                party.members.extend(district_candidates)

                self.nom_msks[nom_party_inds] = True

        else:
            nom_inds = np.random.choice(np.arange(self.N), size=self.nom_rate, replace=False)
            self.nom_msks = np.zeros(self.N).astype(bool)
            self.nom_msks[nom_inds] = True


    def evaluate(self, voting="deterministic", parties=[]):
        candidates = self.residents[self.nom_msks]
        candidate_opis = np.array([candidate.x for candidate in candidates])

        # every resident votes
        resident_opis = np.array([resident.x for resident in self.residents])

        # Single candidate per party (First-past-the-post)
        if voting == "one_per_party":
            candidates = [0] * len(parties)
            candidate_opis = np.full(len(parties), 0).astype(np.float64)

            # Parties nominate one candidate to enter the race
            for i in range(len(parties)):
                pid = parties[i].id

                # local party candidates in the district
                pd_candidates = np.array([candidate for candidate in parties[pid].members
                                          if candidate.d_id == self.d_id])

                party_nom_probs = np.ones(len(pd_candidates)) / len(pd_candidates)

                candidates[pid] = np.random.choice(pd_candidates, p=party_nom_probs, size=1)[0]
                candidate_opis[pid] = candidates[pid].x

            candidates = np.array(candidates)

            # residents vote for the closest candidate
            dists = np.abs(np.subtract.outer(resident_opis, candidate_opis))

        # Proportional representation (closed list)
        else:
            party_opis = [party.x for party in parties]
            dists = np.abs(np.subtract.outer(resident_opis, party_opis))


        # re-scale distance from (0,2) -> (1,0)
        reversed_dists = 1 - (dists / 2)
        pref_thresh = 0.875  # if preference differs lower than 0.125*2
        reversed_dists[reversed_dists < pref_thresh] = 0
        # need to check for residents who are too far from every party

        # propensity to not waste vote
        # only applies with more than 2 parties!
        party_num = len(parties)
        k = 1
        # not straight forward for plurality voting with more than 2 parties

        strat_props = 1 / (1 + np.exp(-k * ((1 / self.rep_num) - self.prev_vote_props)))

        vote = np.argmax(reversed_dists * strat_props, axis=1)


    def vote(self, voting="deterministic", parties=[],
             strategic=False, party_filter=False):

        candidates = self.residents[self.nom_msks]
        candidate_opis = np.array([candidate.x for candidate in candidates])

        # every resident votes
        resident_opis = np.array([resident.x for resident in self.residents])

        # if trust_based:
        #     et = np.array([resident.trust for resident in self.residents])
        #
        #     if any(np.isnan(et)):
        #         print('nan count: {}'.format(np.sum(np.isnan(et))))
        #
        #     is_vote = np.random.binomial(n=1, p=et, size=self.N).astype(bool)
        #     is_vote = (is_vote | self.nom_msks) # candidates always vote
        #
        #     self.vote_masks = is_vote
        #
        #     resident_opis = resident_opis[is_vote]

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

            # Strategic voting
            if len(parties) > 2 and strategic:
                dists = np.abs(np.subtract.outer(resident_opis, candidate_opis))
                # Consequence of distant measure:
                # care less about alignment if parties don't perfectly align w/ preference

                # reverse and re-scale distance so the furthest candidate doesn't get voted
                max_dists = np.max(dists, axis=1)
                rescaled_dists = ((max_dists - dists.T) / max_dists).T
                # Consequence: extreme voters have higher tolerance (shallower slope)

                # re-scale distance from (0,2) -> (1,0)
                # reversed_dists = 1 - (dists / 2)
                # pref_thresh = 0.875  # if preference differs lower than 0.125*2
                # reversed_dists[reversed_dists < pref_thresh] = 0
                # need to check for residents who are too far from every party

                # ideal polling
                sincere_vote = np.argmin(dists, axis=1)
                sincere_vote_count = Counter(sincere_vote)
                poll_props = np.array([sincere_vote_count[i] / sincere_vote.shape[0]
                                       for i in range(len(parties))])

                # propensity to not waste vote
                weighted_vote_props = (self.alpha * self.prev_vote_props) + ((1 - self.alpha) * poll_props)
                print('district {}, weighted vote props:'.format(self.d_id), weighted_vote_props)
                # Need to solve for k if to use logistic function
                # strat_props = 1 / (1 + np.exp(-k * ((1 / self.rep_num) - self.prev_vote_props)))
                # 50% vote to guarantee a victory
                win_probs = np.clip((1/0.5) * weighted_vote_props, a_min=None, a_max=1)
                vote = np.argmax(rescaled_dists * win_probs, axis=1)

            # Naive voting
            else:
                # residents vote for the closest candidate
                vote = np.argmin(np.abs(np.subtract.outer(resident_opis, candidate_opis)), axis=1)

            vote_counter = Counter(vote)
            self.prev_vote_props = np.array([vote_counter[i] / vote.shape[0] for i in range(len(parties))])
            print('district {}, vote prop', self.prev_vote_props)

            winner_id = vote_counter.most_common(1)[0][0]

            # mark winning candidate as being elected
            candidates[winner_id].elected = True

            self.elected = [candidates[winner_id]]
            self.cum_elected.extend(self.elected)

            self.elected_party = [winner_id]
            self.cum_elected_party.extend(self.elected_party)

        # Proportional representation (closed list)
        else:
            party_opis = [party.x for party in parties]

            # Strategic voting
            if len(parties) > 2 and strategic:
                dists = np.abs(np.subtract.outer(resident_opis, party_opis))
                # Consequence of distant measure:
                # care less about alignment if parties don't perfectly align w/ preference

                # reverse and re-scale distance so the furthest candidate doesn't get voted
                max_dists = np.max(dists, axis=1)
                rescaled_dists = ((max_dists - dists.T) / max_dists).T
                # Consequence: extreme voters have higher tolerance (shallower slope)

                # re-scale distance from (0,2) -> (1,0)
                # reversed_dists = 1 - (dists / 2)
                # pref_thresh = 0.875  # if preference differs lower than 0.125*2
                # reversed_dists[reversed_dists < pref_thresh] = 0
                # need to check for residents who are too far from every party

                # ideal polling
                sincere_vote = np.argmin(dists, axis=1)
                sincere_vote_count = Counter(sincere_vote)
                poll_props = np.array([sincere_vote_count[i] / sincere_vote.shape[0]
                                       for i in range(len(parties))])

                # propensity to not waste vote
                weighted_vote_props = (self.alpha*self.prev_vote_props) + ((1-self.alpha)*poll_props)

                # strat_props = 1 / (1 + np.exp(-k * (self.prev_vote_props - (1 / self.rep_num))))
                # 1 / number of seats to guarantee a seat
                crit_thresh = 1/self.rep_num
                win_probs = np.clip((1/crit_thresh) * weighted_vote_props, a_min=None, a_max=1)
                vote = np.argmax(rescaled_dists * win_probs, axis=1)

            # Naive voting
            else:
                # residents vote for the closest candidate
                vote = np.argmin(np.abs(np.subtract.outer(resident_opis, party_opis)), axis=1)

            vote_counter = Counter(vote)
            self.prev_vote_props = np.array([vote_counter[i] / vote.shape[0] for i in range(len(parties))])

            # residents vote for affiliated party
            # vote = np.array([resident.party_aff for resident in self.residents])
            # vote_counter = Counter(vote)

            # Calculate seats for each party (Hamilton's method)
            # Previously d'Hondt method
            party_rep_nums = {}
            a_ratio = {} # advantage ratio
            for pid, vote_count in vote_counter.items():
                party_rep_nums[pid] = 0
                a_ratio[pid] = vote_count

            while (sum(party_rep_nums.values())) < self.rep_num:
                max_a = max(a_ratio.values())
                next_seat = list(a_ratio.keys())[list(a_ratio.values()).index(max_a)]
                party_rep_nums[next_seat] += 1

                a_ratio[next_seat] = vote_counter[next_seat]/(party_rep_nums[next_seat] + 1)

            # print('Seats allocation: {}'.format(party_rep_nums))

            self.elected = []
            self.elected_party = []

            # Parties select their candidates from their party pool to occupy the seats
            for (pid, elected_num) in party_rep_nums.items():

                # local party candidates
                p_candidates = np.array([candidate for candidate in parties[pid].members
                                         if candidate.d_id == self.d_id])
                # if candidate.elected == False (check when party candidates are pooled globally)

                # print('Party {} has {} candidates'.format(pid, p_candidates.shape[0]))

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
    Residents updating their electoral trust (voting probability) based on
    the difference in their policy preference and that of elected officials
    (a possible extension is to include diverse update rules) 
    '''
    def appraise(self, appraise_target, elected_pool=None):

        # Changed from calculating distance to an average position to calculating average absolute distance
        if appraise_target != 'close_global':
            if appraise_target == 'local':
                elected_pool = self.elected

            # Average position of the representatives
            elected_positions = np.array([elected.x for elected in elected_pool])

            resident_opis = np.array([resident.x for resident in self.residents])

            # re-scale distance from (0,2) -> (0,1)
            opi_diffs = np.mean(np.abs(np.subtract.outer(resident_opis, elected_positions)), axis=1) / 2
            # print(opi_diffs)

            # Electoral trust
            et = np.array([resident.trust for resident in self.residents])

            # alpha determines the strength of trust update
            min_alpha = 0.05 # 5% update minimum
            alpha = np.maximum(0.5 - np.abs(0.5 - et), min_alpha) # degree of change (more extreme trust update less)

            # if preference differs lower than 0.125*2, increase trust and lower trust otherwise
            change_et = 0.125 - opi_diffs

            new_et = np.maximum(np.minimum(et + (alpha*change_et), 1), 0) # clip values at (0,1)

            # if any(np.isnan(new_et)):
            #     print('Elected position = {}'.format(elected_position))
            #     print(len(elected_pool))

            for i, resident in enumerate(self.residents):
                resident.trust = new_et[i]

        else:
            # Resident compare their opinions to the closest representative (proportional representation)

            # positions of the representatives
            elected_positions = np.array([elected.x for elected in elected_pool])

            resident_opis = np.array([resident.x for resident in self.residents])

            # re-scale distance from (0,2) -> (0,1)
            close_opidiffs = np.min(np.abs(np.subtract.outer(resident_opis, elected_positions)), axis=1) / 2

            # Electoral trust
            et = np.array([resident.trust for resident in self.residents])

            # alpha determines the strength of trust update
            min_alpha = 0.05  # 5% update minimum
            alpha = np.maximum(0.5 - np.abs(0.5 - et),
                               min_alpha)  # degree of change (more extreme trust update less)

            # if preference differs lower than 0.125*2, increase trust and lower trust otherwise
            change_et = 0.125 - close_opidiffs

            new_et = np.maximum(np.minimum(et + (alpha * change_et), 1), 0)  # clip values at (0,1)

            # if any(np.isnan(new_et)):
            #     print('Elected position = {}'.format(elected_position))
            #     print(len(elected_pool))

            for i, resident in enumerate(self.residents):
                resident.trust = new_et[i]

