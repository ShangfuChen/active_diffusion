from dataclasses import dataclass

@dataclass
class QueryConfig:
    # type of feedback to use (human or ai)
    feedback_agent : str = "human"
    # query method
    query_algorithm : str = "random"
    # number of feedbacks per query
    n_feedbacks_per_query : int = 10