from district import District
from party import Party
import numpy as np
from scipy.spatial import distance
from collections import Counter
import random

class Election:
    """
    Represents an electoral system
    """

    def __init__(self, N, nom_rate = 5, rep_num = 1,
                 party_num = None, party_sd = 0.2, party_loc = 'polarized',
                 district_num = 1, voting='deterministic',
                 opinion_dist_dict = {'dist': "uniform", 'low': -1, 'up': 1},
                 ideo_sort = 0, strategic = False, alpha = 0.5, beta = 0.5):
        """
        Initializes the election model.

        Parameters
        ----------
        N : number of residents (agents) of the entire population
        nom_rate: nomination rate (the rate at which residents become political candidates)
        rep_num: number of representatives
        party_num: number of parties
        party_sd: inclusiveness of parties determining the width of gaussian filter for each party
        party_loc: determine how the positions of parties are assigned
        district_num: number of district
        voting: voting scenario including "deterministic", "probabilistic", "one_per_party", "proportional_rep"
        opinion_dist_dict: distribution of opinions of the district residents
        ideo_sort: the degree of ideological geographic sorting across multiple districts
        strategic: whether residents vote strategically based on past outcome and polling results
        alpha: history bias
        beta: strategic tendency
        """
        self.voting = voting
        self.opinion_dist_dict = opinion_dist_dict
        self.strategic = strategic
        self.alpha = alpha
        self.beta = beta
        self.ideo_sort = ideo_sort

        self.districts = []
        Nd = int(np.round(N / district_num)) # number of residents per district

        # make sure to enough candidates for the number of seats
        if rep_num > nom_rate:
            nom_rate = rep_num + 1

        meta_residents = []
        if district_num > 1:
            if opinion_dist_dict['dist'] == 'uniform':
                gap_size = 2 / district_num  # preference space absolute length = 2
                start = -1000 # scaled up by 1000 to calculate as integer
                stop = 1000
                jump = int(gap_size * 1000)

                # district ideological bound for uniform distribution
                opinion_dist_list = [{'dist': 'uniform', 'low': low / 1000,'up': (low + jump) / 1000}
                                     for low in range(start, stop, jump)]
            elif opinion_dist_dict['dist'] == 'gaussian':
                gap_size = 2 / district_num  # preference space absolute length = 2
                start = int((-1 + (gap_size / 2)) * 1000)  # scaled up by 1000 to calculate as integer
                stop = 1000
                jump = int(gap_size * 1000)
                gaus_sd = opinion_dist_dict['sd']

                # district positions evenly spaced out
                opinion_dist_list = [{'dist': 'gaussian', 'mu': pos / 1000, 'sd': gaus_sd / np.sqrt(district_num)}
                                     for pos in range(start, stop, jump)]

            for i in range(district_num):
                district = District(i, Nd, party_num, nom_rate,
                                    rep_num, alpha, beta,
                                    opinion_dist_list[i])
                self.districts.append(district)

            # ideological sorting
            if ideo_sort < 1:
                movers = []

                # sample a proportion of residents inversely proportional to ideo_sort
                for i in range(district_num):
                    mover_num = int(district.N * (1 - ideo_sort))
                    movers.extend(np.random.choice(self.districts[i].residents,
                                                   size=mover_num, replace=False))

                # shuffling preferences of "movers" and reassign their preference
                mover_prefs = [mover.x for mover in movers]
                random.shuffle(mover_prefs)
                for i, mover in enumerate(movers):
                    mover.x = mover_prefs[i]
        else:
            district = District(0, Nd, party_num, nom_rate,
                                rep_num, alpha, beta,
                                opinion_dist_dict)
            self.districts.append(district)

        if (party_num != None) and (party_num > 0):
            # creating parties based on residents' values and parties try to be distinct
            if party_loc == 'polarized':
                gap_size = 2 / party_num # preference space absolute length = 2
                start = int((-1 + gap_size/2)*1000)
                stop = 1000
                jump = int(gap_size*1000)
                party_pos = [pos / 1000 for pos in range(start, stop, jump)] # party positions evenly spaced out
            elif isinstance(party_loc, list):
                # party positions are manually picked
                party_pos = party_loc
            else:
                # creating parties randomly
                party_pos = np.random.uniform(-1, 1, size=party_num)

            self.parties = [Party(i, party_pos[i], party_sd) for i in range(party_num)]
            self.party_sd = party_sd

            self.affiliate_party() # affiliate residents to each party
        else:
            self.parties = []

        self.elected_pool = []
        self.elected_party_pool = []

        self.cum_elected_pool = []
        self.cum_elected_party_pool = []

        self.model_reporter = {'party_num': lambda m: len(m.parties),
                               'district_num': lambda m: len(m.districts),
                               'rep_num': lambda m: m.districts[0].rep_num,
                               'voting': lambda m: m.voting,
                               'distribution': lambda m: m.opinion_distribution,
                               'js_distance': lambda m: m.position_dissimilarity(),
                               'avg_close_elected': lambda m: m.agg_mean_close_distance()}

        self.step_reporter = {'party_num': lambda m: len(m.parties),
                              'district_num': lambda m: len(m.districts),
                              'rep_num': lambda m: m.districts[0].rep_num,
                              'voting': lambda m: m.voting,
                              'vote_prop': lambda m: m.party_vote_prop(),
                              'seat_prop': lambda m: m.party_seat_prop(),
                              'avg_close_elected': lambda m: m.mean_close_distance()}

    def mean_close_distance(self):
        '''
        Calculate the average distance to closest elected candidates for every resident in every district
        :return: Average distance to closest elected candidates
        '''

        elected_opis = np.array([elected.x for elected in self.elected_pool])
        resident_opis = []

        for i in range(len(self.districts)):
            district = self.districts[i]
            resident_opis.extend([resident.x for resident in district.residents])

        resident_opis = np.array(resident_opis)

        avg_dist = np.mean(np.min(np.abs(np.subtract.outer(resident_opis, elected_opis)), axis=1))

        return avg_dist

    def mean_trust(self):
        '''
        Calculate the average values of trust across the meta population
        :return: Average trust of all agents
        '''

        resident_trusts = []

        for i in range(len(self.districts)):
            district = self.districts[i]
            resident_trusts.extend([resident.trust for resident in district.residents])

        return np.mean(resident_trusts)

    def position_dissimilarity(self, start=0, end=100):
        '''
        Calculate Jensen-Shannon divergence between residents' opinions and elected officials positions
        :return: Distributional distance
        '''

        elected_opis = np.array([elected.x for elected in self.cum_elected_pool])
        resident_opis = []

        for i in range(len(self.districts)):
            district = self.districts[i]
            resident_opis.extend([resident.x for resident in district.residents])

        resident_opis = np.array(resident_opis)

        res_hists = np.histogram(resident_opis, bins=100, range=(-1, 1))[0]
        res_hists = res_hists / res_hists.sum()
        elect_hists = np.histogram(elected_opis, bins=100, range=(-1, 1))[0]
        elect_hists = elect_hists / elect_hists.sum()

        return distance.jensenshannon(res_hists, elect_hists)

    def party_vote_prop(self):
        '''
        Reporting vote proportion that each party receives in a specific electoral cycle
        :return: Vote proportion each party receives in an array
        '''
        tmp_list = []

        for district in self.districts:
            prev_votes = district.prev_vote_props
            tmp_list.append(prev_votes)

        mean_vote_props = np.mean(tmp_list, axis=0)

        return mean_vote_props

    def party_seat_prop(self):
        '''
        Reporting seat proportion that each party wins in a specific electoral cycle
        :return: Seat proportion each party wins in an array
        '''
        tmp_list = []

        for district in self.districts:
            elected_parties = district.elected_party
            tmp_list.extend(elected_parties)

        seat_counter = Counter(tmp_list)
        seat_props = np.array([seat_counter[p_id] for p_id in range(len(seat_counter))])
        seat_props = seat_props / seat_props.sum()

        return seat_props

    def agg_mean_close_distance(self):
        '''
        Calculate the average distance to closest accumulated elected candidates from every district
        :return: Average distance to closest elected candidates
        '''

        elected_opis = np.array([elected.x for elected in self.cum_elected_pool])
        resident_opis = []

        for i in range(len(self.districts)):
            district = self.districts[i]
            resident_opis.extend([resident.x for resident in district.residents])

        resident_opis = np.array(resident_opis)

        avg_dist = np.mean(np.min(np.abs(np.subtract.outer(resident_opis, elected_opis)), axis=1))

        return avg_dist

    def affiliate_party(self):
        for district in self.districts:
            d_residents = district.residents
            resident_opis = np.array([resident.x for resident in district.residents])

            party_opis = [party.x for party in self.parties]

            # residents affiliate with the closest party
            aff = np.argmin(np.abs(np.subtract.outer(resident_opis, party_opis)), axis=1)

            for ind, resident in enumerate(d_residents):
                resident.party_aff = aff[ind]

    def form_new_party(self, party_pos):
        # adding a new party
        self.parties.append(Party(len(self.parties), party_pos, self.party_sd))

        for district in self.districts:
            district.prev_vote_props = np.append(district.prev_vote_props, 0)

    def step(self):

        # reset candidate pools for the new election cycle
        self.elected_pool = []
        self.elected_party_pool = []

        for district in self.districts:
            district.nominate(self.parties)
            district.vote(voting=self.voting, parties=self.parties,
                          strategic=self.strategic)

            # current elected representative pool
            self.elected_pool.extend(district.elected)
            if len(self.parties) > 0:
                self.elected_party_pool.extend(district.elected_party)

        # cumulative elected representative pool
        self.cum_elected_pool.extend(self.elected_pool)
        if len(self.parties) > 0:
            self.cum_elected_party_pool.extend(self.elected_party_pool)

        # electoral feedback updating residents' electoral trust
        # if self.efeedback:
        #     if (self.voting == 'deterministic') or (self.voting == 'probabilistic'):
        #         appraise_target = 'global'
        #         for district in self.districts:
        #             district.appraise(appraise_target, self.elected_pool)
        #     elif (self.voting == 'one_per_party'):
        #         appraise_target = 'local'
        #         for district in self.districts:
        #             district.appraise(appraise_target, None)
        #     elif (self.voting == 'proportional_rep'):
        #         appraise_target = 'close_global'
        #         for district in self.districts:
        #             district.appraise(appraise_target, self.elected_pool)

        # reset candidates after an election
        for party in self.parties:
            party.members = []

    def report_step(self):
        datacollector = {}
        for key in self.step_reporter.keys():
            datacollector[key] = self.step_reporter[key](self)

        return datacollector

    def report_model(self, keys):
        datacollector = {}
        for key in keys:
            datacollector[key] = self.model_reporter[key](self)

        return datacollector

    def __repr__(self):
        """
        Text representation of the model.
        """
        # f'population {variable}' alternative format
        return 'Population (districts: {}| party: {}| electoral system: {}|' +\
         'history-bias: {}| strategic tendency: {})'.format(len(self.districts), len(self.parties),
                                                            self.voting, self.alpha, self.beta)