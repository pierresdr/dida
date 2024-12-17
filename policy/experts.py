    
import os

def load_expert_policy(undelayed_env,persistence,stoch_mdp_distrib,stoch_mdp_param):
    algo_expert = algo_expert.lower()
    expert_loader = eval(algo_expert.upper())
    suffix = ""
    if persistence > 1:
        suffix = "_pers{}".format(persistence)
    if stoch_mdp_distrib is not None:
        suffix = "_noise_{}_param_{}".format(
            stoch_mdp_distrib,
            stoch_mdp_param,
        )
    expert_policy = expert_loader.load(
        os.path.join(
            "trained_agent",
            "{}_{}{}".format(algo_expert, undelayed_env.unwrapped.spec.id, suffix),
            "policy",
        )
    )
    return expert_policy