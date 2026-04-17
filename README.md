# RoboTeach: Online LLM-Automated Curriculum Learning

## Motivation
I hate tuning RL, I hate trying and checking reward functions, I hate all these little tricks and hacks you have to do to get it to work.

It's a lot of just googling for tricks online. What if we can just have an LLM automate this?

## Method:
- Have a generic callback that tracks if our train reward is plateauing
- If we plateau, fire the LLM curriculum generator
- Record a few rollouts and pass them to the LLM
- Ask the LLM to generate a reward function
- Modify reward function online via Gymnasium wrapper
- Continue training

## Implementation
`callback.py`: the callback that tracks when to trigger the new curriculum. Also contains a callback that passes video to the LLM
`agent.py`: the LLM agent that takes in video and returns a new reward function as a set of weights
`wrapper.py`: a wrapper around the Gymnasium environment that allows online modification of the reward function
`train.py`: the orchestration of the whole system
