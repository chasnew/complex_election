from district import District
from party import Party
import numpy as np
from collections import Counter

class Election:
    """
    Represents an electoral system
    """

    def __init__(self, N, nom_rate = 0.05, rep_num = 1,
                 party_num = None, party_sd = 0.2,
                 district_num = 1, voting='deterministic',
                 opinion_distribution = "uniform"):
        """
        Initializes the election model.

        Parameters
        ----------
        N : number of residents (agents) in each district
        nom_rate: nomination rate (the rate at which residents become political candidates)
        rep_num: number of representatives
        """
        self.voting = voting

        self.districts = []
        for i in range(district_num):
            district = District(N, nom_rate, rep_num,
                                opinion_distribution)
            self.districts.append(district)

        if (party_num != None) and (party_num > 0):
            party_pos = np.random.uniform(-1, 1, size=party_num)
            self.parties = [Party(i, party_pos[i], party_sd) for i in range(party_num)]
            self.party_sd = party_sd
        else:
            self.parties = None

        self.elected_pool = []

    def step(self):

        self.elected_pool = []

        for district in self.districts:
            district.nominate(self.parties)
            district.vote(voting=self.voting, parties=self.parties)

            self.elected_pool.extend(district.elected)

    def __repr__(self):
        """
        Text representation of the model.
        """
        return f'Population (districts: {len(self.districts)}| per district size: {self.N}| party: {self.party_num})'