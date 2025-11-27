# Training MVP Todo:

- Make a verification directory to add all the action verification code to. Use this to help quickly create action masks.
- Add subtype enum to all card prefabs (i think only IKZ cards do not have a subtype)

## Training 

- Remove masking (and unit2 bin2 branching) from the action decode forward pass. Instead pass the mask (and head2 paths) to pufferlib sampler, and do sequential masking on each head in the sampling step during training.

https://chatgpt.com/c/691d7f1b-be34-832e-95c9-6fd3ac85268b

```
• Looks like the masking logic is indeed broken before you ever get to the sub‑action filters.

  - python/src/policy/tcg_policy.py:420 correctly pulls primary_action_mask =
    action_mask_struct["primary_action_mask"].to(dtype=torch.bool) from the observation, but the very next thing
    that happens is primary_action_mask = torch.zeros(B, 13, ...) at python/src/policy/tcg_policy.py:431. That second
    (line 465) forces every logit to -1e9, so the softmax is uniform and chosen_actions inside the policy always
    degenerates to index 0 (ACT_NOOP). The environment later clamps the sampled primary back to something legal
    using its own mask, which is why you still see a “set” primary in your logs, but inside the policy we only
    ever build masks for primary 0, so every sub‑action mask (legal_subaction1_mask, legal_subaction2_mask,
    legal_subaction3_mask) ends up empty and the logits for those components get fully masked, leaving 0 as the only
    surviving index (python/src/policy/tcg_policy.py:475‑521). That perfectly matches the [*, 0, 0, 0] traces and the
    eventual “invalid action” errors once the env tries to execute those tuples.
  - Even after you fix the line above, there’s another conceptual issue: the sub‑action masks/logits are conditioned
    on chosen_actions = primary_action_probs.argmax(dim=-1) (lines 465‑473). The actual action that is sent to the
    environment isn’t that argmax; it is sampled later from the logits (or even clamped in _enforce_action_mask).
    So as soon as exploration causes the sampler to pick a different primary than the argmax, the sub‑action logits
    no longer refer to the same combination and you again end up with masks that don’t match the action that will be
    executed. You either need to plumb the sampled primary back into the policy before building the sub‑action masks,
    or (more robustly) treat the enumerated legal_actions list as the thing you sample from so you never lose the
    joint legality information.

  So in short: yes, there is a concrete bug—the primary mask gets nulled out—which in turn explains why subaction
  masks are empty and the model emits [*, 0, 0, 0]. Even once that is fixed you’ll still need to rethink how the sub-
  actions are conditioned so they correspond to the primary that will actually be played.


› Okay i fixed the issue with the torch.zeros line. But for the action heads being able to be masked and handle
  exploration. What do you think some good options are? Would a single action head outputting ints from 1000 -> 9999
  and somehow also handling noop 0000 work? Or keep the multiple action heads and somehow handle the specific valid
  combinations while still being able to handle exploration?


• I’m putting together thorough suggestions on different architectural approaches for sampling and masking in multi-
  head setups, referencing specific lines in python/src/policy/tcg_policy.py. I’ll outline options like single
  categorical sampling over legal actions, embedding-based approaches, hierarchical sequential sampling, joint mask
  restructuring, and controlled exploration using methods like Gumbel top-k, connecting each to constraints from
  the docs.

─ Worked for 9s ─────────────────────────────────────────────────────────────────────────────────────────────────────

• Masking Strategy

  - python/src/policy/tcg_policy.py:420-475 still assumes it can recombine per-field masks after the fact. Even with
    the zeroing bug fixed, chosen_actions = primary_action_probs.argmax(dim=-1) (line 472) decouples the subaction
    sampling from whatever primary gets drawn after exploration, so the masks for subaction_1..3 end up describing a
    different action tuple than the one you actually dispatch.

  Options Going Forward
    ε-greedy, etc.—and any sample you draw is guaranteed legal. Complexity moves into embedding the (primary, sub1,
    sub2, sub3) tuple for the policy head, but that’s straightforward.
  - 2) Keep the multi-head layout but condition on actual samples. After you sample a primary from the logits (or
    apply ε-greedy, Gumbel-Top-k, etc.), feed that sampled index back into the masking logic before you compute
› Is it possible to do autoregressive sampling? Would i just need to change the /ext/SkyPufferLib code or is it
  possible to do without doing that? (I don't mind changing the code btw). But doing some research, it appears that
  for OpenAI Five which im emulating, sampling during training was sequential. Where each sampled head had a mask
  that depended on the previous. So in this instance, there are four separate action heads, the forward pass is a
  single pass, but the sampling is a sequential factorization. Is that possible? Would I just need to update the
  pufferl.py code?


• Explored
  └ List python
    List python
    List policy
    List ext
    Search SkyPuffer
    List ext
    Search autoregressive in SkyPufferLib
    List SkyPufferLib
    Read pufferl.py
    Search pufferl in pufferlib
    Read pytorch.py
    Search sample_logits
    Search sample_logits in SkyPufferLib
    Search sample_logits in SkyPufferLib
    Read pufferl.py
    Search mb_actions in pufferl.py
    Read pufferl.py

─ Worked for 2m 14s ─────────────────────────────────────────────────────────────────────────────────────────────────

• - In the current stack the sampling logic lives in pufferlib.pytorch.sample_logits (/ext/SkyPufferLib/pufferlib/
    pytorch.py:189-226). Whenever PuffeRL needs an action—both while gathering rollouts (/ext/SkyPufferLib/pufferlib/
    touch pufferl.py itself. To get autoregressive sampling you only need to teach sample_logits (or a replacement
    you register there) how to:
      1. Draw the primary action;
      2. Recompute/mask the logits for subaction‑1 given that primary;
         The easiest way to do this is to have your policy return a custom container with everything sample_logits
         needs (e.g., unconditioned logits for each head plus a method that, given a prefix, spits out the masked
         logits of the next head). Then extend sample_logits with a new branch that recognizes that container type
         and performs the sequential sampling. As long as the same branch is used when action is provided (the PPO
         update path), your log‑prob/entropy math stays correct.
  - If you would rather avoid editing /ext/SkyPufferLib, the alternative is to stop relying on sample_logits
    entirely: make your policy produce a single categorical over the enumerated legal tuples, sample inside the
    policy, and write the resulting (primary, sub1, sub2, sub3) into the action tensor you return. You would
    also emit the scalar log‑prob/entropy alongside it so the PPO loss can be computed. That approach sidesteps
    autoregressive logic, but it means your policy has to maintain the mapping from “flattened tuple id” back to the
    MultiDiscrete action vector expected by the env.
  - If you specifically want OpenAI‑Five style sequential heads, patching pufferlib.pytorch.sample_logits is the
    most direct route. You don’t have to change pufferl.py; just make sure your policy exposes enough information
    for the sampler to re-mask subheads given previously sampled values, and update sample_logits so it a) runs the
    sequential factorization during rollout, and b) when the PPO update calls it with action=..., it walks through
    the same sequence deterministically to reconstruct the log‑prob for that stored action. Once those two pieces are
    in place, exploration works exactly like the Five papers describe.


› Awesome. So applying masking this one sequentially in the sampling will not break PPO and allows for proper
  computation and exploration? I'm also curious about another issue. Right now, subaction_2 head has a thing where
  it takes different computational paths depending on if the head needs to target a unit or not. Sometimes the head
  is used to target a unit and other times it's just an index so it doesn't need the unit attention part (i think?).
  How is that handled in this scenario since which logits returned were originally dependent on which action was
  taken? Or would it be okay to just always have that head use units as attention?


• I'm working through how sequential sampling aligns with PPO, emphasizing that logprob accumulation must follow the
  same sequential logic during both sampling and evaluation to keep gradients consistent. For subaction2 gating, I
  see that decisions depend on the chosen primary action, which can be sampled sequentially to determine which logits
  (unit or bins) to use next. The approach involves structuring outputs to provide both logits sets and masks, then
  selecting based on the primary action sampled. This keeps sampling consistent with gating logic, avoiding semantic
  mismatches. I also plan to reference relevant code lines in python/src/policy/tcg_policy.py (around lines 465 and
  504-508) to ground the explanation and give practical suggestions. Let's finalize a friendly, concise response with
  bullet points linking these ideas.

─ Worked for 13s ────────────────────────────────────────────────────────────────────────────────────────────────────

• - Autoregressive sampling works cleanly with PPO as long as sample_logits (or whatever sampler you plug into
    pufferlib) performs both the sequential draw and the log-prob reconstruction when it’s called with stored actions
    during the PPO update. In other words, after you sample primary → sub1 → sub2 → sub3 while collecting rollouts,
    the same helper must walk through the exact same conditional masks when PPO later calls it with action=... to
    recompute logprob (/ext/SkyPufferLib/pufferlib/pufferl.py:272 for rollout, pufferl.py:386 for PPO). Do that and
    the policy gradient math stays valid, so exploration through ε-greedy/temperature/etc. is unaffected.
  - For the subaction‑2 split path: right now python/src/policy/tcg_policy.py:504-510 wires requires_unit_subaction2
    = (chosen_actions == ACT_ATTACK) | (chosen_actions == ACT_ATTACH_WEAPON_FROM_HAND) and uses torch.where to pick
    either unit2_logits (attention over the target matrix) or bins2_projected_hidden (flat index logits). Once you
    pivot to sequential sampling, you should reuse the same structure except driven by the sampled primary, not the
    greedy argmax. After sampling the primary you already know, per batch element, whether that action needs a unit
    target or a plain index; apply the gating before sampling subaction‑2 so the logits/masks line up. There’s no
    need to “always use the unit attention” — those “bin” logits exist precisely because a lot of actions (e.g., pick
    one of M enum bins, or toggle IKZ token use) don’t map onto the hand/entity embedding set. Forcing them through
    the unit attention head would give meaningless scores for non-entity decisions and still wouldn’t fix the mask
    mismatch.
  - Implementation sketch: keep both candidate logits (unit2_logits, bins2_logits), but when the sampler processes
    step 2 and sees “this batch entry chose ACT_ATTACK,” it slices the unit logits, masks them with the legal tuples
    for that branch, samples a target, and adds the resulting log-prob to the running total. For entries whose
    sampled primary doesn’t require a unit, it instead uses the precomputed bin logits and associated masks. This
    lets the same head support both behaviors while staying deterministic; the PPO update will replay exactly the
    same branch for each stored action, so log-probs remain correct.

  Next steps

  1. Extend pufferlib.pytorch.sample_logits (or a wrapper) with a branch that understands your policy’s
     autoregressive output and performs sequential sampling + log-prob accumulation.
  2. Refactor decode_actions so it exposes the logits/masks for each conditional head given a prefix; after sampling
     primary/sub1 you simply request the next head with that prefix, keeping the unit/bin gating (python/src/policy/
     tcg_policy.py:504-510) but keyed on the sampled value.
  3. Verify by logging sampled tuples and the legal_actions list to ensure every sampled action is one of the
     enumerated legal combinations; then run PPO to confirm the KL/entropy terms remain finite.
```
