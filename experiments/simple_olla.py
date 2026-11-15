"""Standard negative-feedback OLLA; PHY threshold map is provided externally."""
from dataclasses import dataclass
import numpy as np

@dataclass
class SimpleOLLA:
    SINR_offset:float=0.
    minimum_db:float=-20.
    maximum_db:float=20.
    @property
    def offset_db(self):return self.SINR_offset
    def update(self,ack):
        if type(ack) is not bool:raise TypeError('OLLA expects one actual bool ACK/NACK')
        self.SINR_offset+=-.1 if ack else .9
        self.SINR_offset=float(np.clip(self.SINR_offset,self.minimum_db,self.maximum_db))
        return self.SINR_offset
    def select_mcs(self,measured_sinr_db,threshold_table):
        effective=float(measured_sinr_db)-self.SINR_offset
        eligible=np.flatnonzero(np.asarray(threshold_table)<=effective)
        return int(eligible[-1]) if len(eligible) else 0
