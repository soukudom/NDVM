from abc import ABC, abstractmethod

class AbstractMetric(ABC):
    """Abstract base class metric.
    """

    def __init__(self,dataset,label,multiclass,verbose):
        pass

    @abstractmethod
    def run_evaluation(self):
        raise NotImplementedError

    @abstractmethod
    def get_details(self, output_dir_metadata_base):
        raise NotImplementedError
        
