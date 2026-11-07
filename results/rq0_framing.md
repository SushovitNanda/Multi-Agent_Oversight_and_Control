# RQ0: Is causal closure an invariant of honest multi-agent activity?

## The question

Every log-based oversight method for agent swarms rests on one unstated premise: the logs are
causally complete. A monitor, an LLM judge, a provenance graph, a control protocol with a trusted
overseer: each one reads records and assumes that what an agent knows or does can be traced to
something that was recorded. The sprint's own motivating incident breaks exactly this premise.
About 1,200 agents found a shared directory that nobody was logging as a channel, turned it into
a message board, and about 7% went on to spoof or delete their transcripts.

So before asking how to catch liars (RQ1, RQ2), hidden channels (RQ3), partial views (RQ4) or
informed adversaries (RQ5), there is a prior question:

> **RQ0.** In an honest swarm, does everything an agent comes to know and do have a logged
> cause? If closure holds, then "nothing comes from nowhere" is an invariant that a liar must
> break, and oversight can be built on breaks of the invariant rather than on reading content.
> If closure fails routinely in honest swarms, then no log-based audit can have both a low
> false-alarm rate and any teeth, however good its judge is.

This is fundamental in the sense that it affects a lot of other problems at once. It is the
condition under which transcript monitors, judge models, replica-disagreement checks, provenance
systems and AI-control protocols are even well posed for a population of agents. It also tells an
operator what has to be logged: the channels that honest knowledge actually travels on.

## How RQ0 is answered here

Three sources of evidence, two of them already in hand.

1. **Honest simulated swarms, three model families, truthful logs.** 60 honest episodes on
   deepseek-v4-flash and 15 each on gemma-4-31b-it and qwen3.8-27b, 12 agents, 20 rounds,
   planted tracers. Measured: alert rate under the four closure rules with every sanctioned
   channel modelled; how often agents handle another agent's private fact at all (benign
   forwarding); and which channel explains each such sighting (message, shared file, none).
   Table: `results/rq0.csv`.

2. **The two channels the auditor initially failed to model.** The first audit pass over the
   180-episode main run raised three alerts on untampered honest-labelled episodes. Both causes
   were missing channels, not misbehaving agents: a message sent in the final round can never be
   received (delivery happens at the next round start), and a tracer can travel through a shared
   file that one agent writes and another reads. Once those two channels were modelled the alerts
   went to zero. `rq0.csv` reports the alert rate with and without each channel, which is the
   measured cost of an incomplete channel model.

3. **A real swarm.** AI Village swarm-village (133 agents, research terms, local only). Natural
   tracers are rare tokens that appear in exactly one agent's private activity (computer-use
   actions and tool outputs, plus the agent's private memory notes in the extended run) before
   any public chat. A transfer "from nowhere" is a later public message by a different agent
   containing that token with no earlier public message carrying it. Result on the base run:
   28,752 single-owner tracers, 50 later appeared in chat with a visible path, 0 without.
   Extended run with memories: see `results/ai_village_tracers_swarm-village_with_memories.json`.

## Measured (truthful logs, all channels modelled unless stated)

| Family, condition | n | alert, full model | alert, no file edges | alert, no final-round rule | non-owner sightings / episode | explained by messages | needs file hop | unexplained |
|---|---|---|---|---|---|---|---|---|
| deepseek, honest | 60 | 0.000 | 0.000 | 0.000 | 0.0 | - | - | - |
| deepseek, in-band | 60 | 0.000 | 0.050 | 0.100 | 8.2 | 0.99 | 0.01 | 0.00 |
| deepseek, off-log | 60 | 0.933 | 0.933 | 0.933 | 6.5 | 0.14 | 0.00 | 0.86 |
| gemma, honest | 15 | 0.000 | 0.000 | 0.200 | 0.0 | - | - | - |
| gemma, in-band | 15 | 0.000 | 0.000 | 0.067 | 3.1 | 1.00 | 0.00 | 0.00 |
| gemma, off-log | 15 | 0.533 | 0.533 | 0.667 | 4.0 | 0.62 | 0.00 | 0.38 |
| qwen, honest | 15 | 0.000 | 0.000 | 0.000 | 0.0 | - | - | - |
| qwen, in-band | 15 | 0.000 | 0.000 | 0.067 | 0.1 | 1.00 | 0.00 | 0.00 |
| qwen, off-log | 15 | 0.067 | 0.067 | 0.067 | 0.1 | 0.00 | 0.00 | 1.00 |

Real swarm (AI Village swarm-village, computer-use turns plus private memories as private
activity): 32,033 single-owner tracers, 19 later appeared in public chat with a visible path,
0 without. Base run (turns only): 28,752 tracers, 50 with a path, 0 without.

Reading. Honest agents in this testbed never handle another agent's private fact at all, so
closure holds trivially for them. The in-band collusive episodes are the informative case for
the premise: agents pass private facts to each other over sanctioned channels about eight times
per episode on deepseek, and every one of those transfers has a logged path once messages and
shared files are both modelled. Leave out shared files and 5% of these episodes false-alarm;
leave out the final-round rule and 10% do (20% of gemma honest episodes). Closure breaks only
where the hidden channel was used: 86% of non-owner sightings are unexplained in deepseek
off-log episodes, 38% in gemma (whose colluders also leak in-band, giving most tracers a
logged path), and the single qwen off-log sighting is unexplained. In the real swarm, no
private fact reached public chat without a visible path.

## What a yes and a no would mean

- **Closure holds (alert rate 0 on honest logs, 0 unexplained real-world transfers):** honest
  activity never breaks the invariant, so every break is evidence. RQ1 to RQ5 then become
  questions about *how* adversaries break it and whether enough honest witnesses remain:
  who can erase what (RQ2, witness theory), what a hidden channel leaves behind (RQ3), which
  views are enough (RQ4), and whether an adversary who knows the rules can act without breaking
  them (RQ5).
- **Closure fails in honest swarms:** the audit's false-alarm floor is set by the honest swarm,
  not by the adversary, and the channels responsible must be logged before any later RQ is
  meaningful. The three false alarms in the first pass are a small instance of this: two
  unmodelled channels, three alerts in 180 episodes, removed by modelling the channels, not by
  weakening the rules.

## Limitations specific to RQ0

- The simulated swarm has a closed world: the only channels are the ones the simulator offers.
  The real-world check covers one village and one kind of private activity, and its tracers are
  heuristic tokens, not planted facts.
- "No logged path" in AI Village can only be measured for public chat; private-to-private
  transfers (a shared web page read by two agents) are invisible to this check by construction.
  The measured 0 is therefore a lower bound on closure violations, reported as such.
