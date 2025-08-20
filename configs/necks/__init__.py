from .gap import GlobalAveragePooling
from .part_gap_SP import PartGlobalAveragePooling_SP
from .hr_fuse import HRFuseScales
from .part_gap_recurrent import PartGlobalAveragePooling_recurrent
from .part_gap_recurrent_no_BP import PartGlobalAveragePooling_recurrent_no_BP
from .part_gap_MSP import PartGlobalAveragePooling_MSP
from .part_gap_MSP_BP1 import PartGlobalAveragePooling_MSP_BP1

__all__ = ['GlobalAveragePooling', 'HRFuseScales', 'PartGlobalAveragePooling_SP', 'PartGlobalAveragePooling_recurrent',
            'PartGlobalAveragePooling_MSP','PartGlobalAveragePooling_MSP_BP1','PartGlobalAveragePooling_recurrent_no_BP',]
