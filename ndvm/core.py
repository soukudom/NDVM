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
    def get_details(self):
        raise NotImplementedError
