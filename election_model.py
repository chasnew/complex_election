from district import District
from party import Party
import numpy as np
from scipy.spatial import distance

class Election:
    """
    Represents an electoral system
    """

    def __init__(self, N, nom_rate = 0.05, rep_num = 1,
                 party_num = None, party_sd = 0.2,
                 district_num = 1, voting='deterministic',
                 opinion_distribution = "uniform",
                 gaussian_mu = 0, gaussian_sd = 0.5):
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
        """
        self.voting = voting
        self.opinion_distribution = opinion_distribution

        self.districts = []
        Nd = int(np.round(N / district_num)) # number of residents per district

        for i in range(district_num):
            district = District(i, Nd, nom_rate, rep_num,
                                opinion_distribution,
                                gaussian_mu, gaussian_sd)
            self.districts.append(district)

        if (party_num != None) and (party_num > 0):
            party_pos = np.random.uniform(-1, 1, size=party_num)
            self.parties = [Party(i, party_pos[i], party_sd) for i in range(party_num)]
            self.party_sd = party_sd
        else:
            self.parties = []

        self.model_reporter = {'party_num': lambda m: len(m.parties),
                               'district_num': lambda m: len(m.districts),
                               'rep_num': lambda m: m.districts[0].rep_num,
                               'voting': lambda m: m.voting,
                               'distribution': lambda m: m.opinion_distribution,
                               'js_distance': lambda m: m.position_dissimilarity()}


        self.elected_pool = []
        self.elected_party_pool = []

        self.cum_elected_pool = []
        self.cum_elected_party_pool = []

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

    def step(self):

        # reset candidate pools for the new election cycle
        self.elected_pool = []
        self.elected_party_pool = []

        for district in self.districts:
            district.nominate(self.parties)
            district.vote(voting=self.voting, parties=self.parties)

            self.elected_pool.extend(district.elected)
            self.elected_party_pool.extend(district.elected_party)

        # cumulative elected representative pool
        self.cum_elected_pool.extend(self.elected_pool)
        self.cum_elected_party_pool(self.elected_party_pool)

        # reset candidates after an election
        for party in self.parties:
            party.members = []

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