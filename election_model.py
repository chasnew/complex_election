from district import District
from party import Party
import numpy as np
from scipy.spatial import distance
import multiprocessing as mp

class Election:
    """
    Represents an electoral system
    """

    def __init__(self, N, nom_rate = 0.05, rep_num = 1,
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
                party_pos = []
                res_inds = np.arange(meta_residents.shape[0])
                meta_opis = np.array([resident.x for resident in meta_residents])
                likelihood_mass = np.ones(meta_residents.shape[0])

                for i in range(party_num):
                    party_probs = likelihood_mass / likelihood_mass.sum()
                    rp_ind = np.random.choice(res_inds, p=party_probs)
                    party_pos.append(meta_residents[rp_ind].x)

                    diff_square = np.square(meta_opis - meta_residents[rp_ind].x)
                    gaussian_filter = np.exp((-diff_square) / (2 * np.square(party_sd)))

                    likelihood_mass = likelihood_mass - gaussian_filter
                    likelihood_mass[likelihood_mass < 0] = 0
                    if all(likelihood_mass == 0):
                        likelihood_mass += 0.1

                party_pos = np.array(party_pos)
                self.parties = [Party(i, party_pos[i], party_sd) for i in range(party_num)]
            else:
                # creating parties randomly
                party_pos = np.random.uniform(-1, 1, size=party_num)
                self.parties = [Party(i, party_pos[i], party_sd) for i in range(party_num)]

            self.party_sd = party_sd
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
            for district in self.districts:
                district.appraise(self.elected_pool)

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