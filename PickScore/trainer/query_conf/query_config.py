from dataclasses import dataclass

@dataclass
class QueryConfig:
    # type of feedback to use (human or ai)
    feedback_agent : str = "human"
    # feedback_agent : str = "ai"
    # query method
    # query_type : str = "perplexity"
    query_type : str = "random"
    # query_type : str = "ensemble_std"
    # query_type : str = "all"

    query_everything_fisrt_iter : bool = False

    # Only used in random query
    n_feedback_per_query : int = 10
    # n_feedback_per_query : int = 20
    # n_feedback_per_query : int = 40

    # Only used in active query methods where number of queries vary in each loop.
    # If not enough queries are chosen, choose queries randomly to meet this minimum requirement    
    min_n_queries : int = 10

    # whether to only enforce min number of queries during warmup 
    only_enforce_min_queries_during_warmup : bool = False

    # Perplexity query threshold
    perplexity_thresh : float = 1.5

    # Ensemble std query threshold
    ensemble_thresh_is_hard : bool = False # use fixed value of thresh ensemble_std_thresh. if false use ensemble_dynamic_std_thresh
    # ensemble_thresh_is_hard : bool = False # use fixed value of thresh ensemble_std_thresh. if false use ensemble_dynamic_std_thresh
    ensemble_std_thresh : float = 1.0
    ensemble_dynamic_std_thresh : float = 0.01 # percentage of range of human rewards seen so far

    # Config for using real human feedback
    use_best_image : bool = True
    