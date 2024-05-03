from dataclasses import dataclass

@dataclass
class QueryConfig:
    # type of feedback to use (human or ai)
    # feedback_agent : str = "human"
    feedback_agent : str = "ai"
    # query method
    query_type : str = "random"
    # query_type : str = "ratio_std"
    # number of feedbacks per query
    n_feedbacks_per_query : int = 10
