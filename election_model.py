from district import District
from party import Party
import numpy as np
from scipy.spatial import distance
import multiprocessing as mp

class Election:
    """
    Represents an electoral system
    """

    def __init__(self, N, nom_rate = 5, rep_num = 1,
                 party_num = None, party_sd = 0.2, polarization = True,
                 district_num = 1, voting='deterministic',
                 opinion_distribution = "uniform",
                 gaussian_mu = 0, gaussian_sd = 0.5,
                 efeedback = False):
        """
        Initializes the election model.

        Parameters
        ----------
        N : number of residents (agents) of the entire population
        nom_rate: nomination rate (the rate at which residents become political candidates)
        rep_num: number of representatives
        party_num: number of parties
        party_sd: inclusiveness of parties determining the width of gaussian filter for each party
        district_num: number of district
        voting: voting scenario including "deterministic", "probabilistic", "one_per_party", "proportional_rep"
        opinion_distribution: distribution of opinions of the district residents
        gaussian_mu: mean of the gaussian distribution of opinions
        gaussian_sd: standard deviation of the gaussian distribution of opinions
        efeedback: whether electoral feedback is activated allowing residents to update their electoral trust
        """
        self.voting = voting
        self.opinion_distribution = opinion_distribution
        self.efeedback = efeedback

        self.districts = []
        Nd = int(np.round(N / district_num)) # number of residents per district

        # make sure to enough candidates for the number of seats
        if rep_num > nom_rate:
            nom_rate = rep_num + 1

        meta_residents = []
        for i in range(district_num):
            district = District(i, Nd, nom_rate, rep_num,
                                opinion_distribution,
                                gaussian_mu, gaussian_sd)
            self.districts.append(district)

            meta_residents.append(district.residents)

        meta_residents = np.concatenate(meta_residents)

        if (party_num != None) and (party_num > 0):
            # creating parties based on residents' values and parties try to be distinct
            if polarization:
                gap_size = 2 / party_num # preference space absolute length = 2
                start = int((-1 + gap_size/2)*1000)
                stop = 1000
                jump = int(gap_size*1000)
                party_pos = [pos for pos in range(start, stop, jump)] # party positions evenly spaced out
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
                               'efeedback': lambda m: m.efeedback,
                               'js_distance': lambda m: m.position_dissimilarity(),
                               'avg_close_elected': lambda m: m.agg_mean_close_distance()}

        self.step_reporter = {'party_num': lambda m: len(m.parties),
                              'district_num': lambda m: len(m.districts),
                              'rep_num': lambda m: m.districts[0].rep_num,
                              'voting': lambda m: m.voting,
                              'avg_trust': lambda m: m.mean_trust(),
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

    def position_dissimilarity(self):
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

    def step(self):

        # reset candidate pools for the new election cycle
        self.elected_pool = []
        self.elected_party_pool = []

        for district in self.districts:
            district.nominate(self.parties)
            district.vote(voting=self.voting, parties=self.parties,
                          trust_based=self.efeedback)

            # current elected representative pool
            self.elected_pool.extend(district.elected)
            if len(self.parties) > 0:
                self.elected_party_pool.extend(district.elected_party)

        # cumulative elected representative pool
        self.cum_elected_pool.extend(self.elected_pool)
        if len(self.parties) > 0:
            self.cum_elected_party_pool.extend(self.elected_party_pool)

        # electoral feedback updating residents' electoral trust
        if self.efeedback:
            if (self.voting == 'deterministic') or (self.voting == 'probabilistic'):
                appraise_target = 'global'
                for district in self.districts:
                    district.appraise(appraise_target, self.elected_pool)
            elif (self.voting == 'one_per_party'):
                appraise_target = 'local'
                for district in self.districts:
                    district.appraise(appraise_target, None)
            elif (self.voting == 'proportional_rep'):
                appraise_target = 'party'
                party_elected_list = []
                for i in range(len(self.parties)):
                    tmp_elected = [candidate for candidate in self.elected_pool if candidate.party_id == i]
                    party_elected_list.append(tmp_elected)

                for district in self.districts:
                    district.appraise(appraise_target, party_elected_list)




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
        return f'Population (districts: {len(self.districts)}| party: {len(self.parties)})'