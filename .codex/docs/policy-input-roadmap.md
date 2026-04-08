# Policy Input Roadmap (TCG v2 Only)

Purpose: track candidate observation/model inputs to add to the v2 policy, based on OpenAI Five style ideas and our design discussion.

Scope:
1. This roadmap applies only to the v2 policy and v2 observation/marshaling path.
2. v1 policy and v1 observation schema remain frozen for backward compatibility with old checkpoints.

## 1) Full Ordered Backlog of Potential New Inputs

1. Previous action tuple (self)
   - `prev_primary`
   - `prev_sub1`
   - `prev_sub2`
   - `prev_sub3`
   - `prev_action_valid` (0 on first decision context)
   - optional: `prev_was_pass`
   - Notes:
     - OpenAI Five analog: previous sampled action.
     - High value with very small footprint.

2. Active combat / response-window attacker context
   - `response_attacker_valid`
   - `response_attacker_slot_index`
   - `response_attacker_is_leader`
   - `response_attacker_card_def_id`
   - `response_target_is_leader`
   - `response_target_slot_index`
   - `response_target_card_def_id`
   - optional: `response_attack_declared_steps_ago`
   - Notes:
     - In defender decision windows, the policy should not need to infer "which opposing entity or leader is attacking right now" purely from recurrent state.
     - This is public information and should be exposed directly whenever combat/response is active.
     - If some of this already exists in combat-context fields elsewhere, treat this item as a roadmap reminder to preserve that signal in the actual v2 marshaled inputs.

3. Opponent previous action tuple / short recent action history
   - Minimum parity block:
     - `opp_prev_primary`
     - `opp_prev_sub1`
     - `opp_prev_sub2`
     - `opp_prev_sub3`
     - `opp_prev_action_valid`
     - optional: `opp_prev_was_pass`
   - Optional expansion:
     - last-`N` opponent action tuples (same schema as self)
     - or compact opponent-action event tokens + pooling
   - Notes:
     - Current self-only previous-action input leaves a tactical blind spot during response windows.
     - In many combat-response situations, the most informative recent opponent action is the attack declaration itself, which identifies the attacking entity or leader.
     - Prefer explicit combat-state attacker fields even if opponent action history is added; action history should complement, not replace, direct public combat context.

4. Effect system upgrade: effect embeddings instead of only fixed booleans
   - Replace/augment fields like `is_frozen`, `is_shocked`, `is_effect_immune` with an effect-set payload per entity.
   - Per-effect fields (for each slot in `effects[K]`):
     - `effect_type_id` (embedding)
     - `duration_kind` (`none | finite | infinite`)
     - `turns_remaining` (0 when infinite)
     - `stack_count`
     - `source_owner` (`self | opponent`)
     - `source_zone` (leader/gate/alley/garden/hand/discard/other)
     - `source_card_def_id` (embedding)
     - `source_entity_index` (index/scalar)
     - `effect_active_now` (bool)
   - Encoding:
     - masked set encoder (same pattern as current zone/weapon set processors).

5. Ability dynamic readiness state
   - For each relevant ability/card:
     - `can_use_now`
     - `turns_until_usable`
     - `cooldown_remaining`
     - `cost_shortfall` (how far from payable)
     - `legal_target_count`
   - Notes:
     - Distinct from static ability metadata (`ability_timing`, `has_ability`, etc.).

6. Board relation/topology features (non-CNN default)
   - Not necessarily a CNN first.
   - Candidate structured relations:
     - `can_attack_matrix[src, dst]`
     - `can_target_matrix[src, dst]`
     - per-unit threat counters:
       - `num_attackable_targets`
       - `num_attackers_threatening_me`
     - lane pressure summaries by zone/index.
   - Notes:
     - Likely better ROI than CNN for current structured board representation.

7. Hidden-information belief state (no information leakage)
   - Global belief/certainty payload candidates:
     - `opp_unknown_hand_slots`
     - `opp_known_hand_slots`
     - `turns_since_last_hand_reveal`
     - `revealed_cards_seen_total`
     - `revealed_cards_unresolved_total`
     - `known_hand_type_hist[card_type_count]`
     - `known_hand_keyword_hist` (e.g. charge/defender/infiltrate/has_ability)
     - `p_hand_type[card_type_count]`
     - `p_next_draw_type[card_type_count]`
     - `expected_playable_cards_now`
     - `p_has_immediate_removal`
     - `p_has_charge_threat`
     - `p_has_defender_access`
     - `expected_attack_power_next_turn`
     - `hand_belief_entropy`
     - `deck_belief_entropy`
     - `belief_confidence`
   - Construction constraints:
     - Only use public evidence (reveals/searches/discards/played cards/public zones).
     - Keep hard facts (`known_*`) separate from probabilistic estimates (`p_*`).
     - Include uncertainty/confidence channels so policy can down-weight weak beliefs.
   - Reveal-memory recommendation:
     - Split reveal handling into two separate feature blocks:
       - `current_hidden_belief`: "cards the opponent may still be holding right now".
       - `reveal_history_memory`: "cards/archetypes we have seen at any point this game".
     - Do not collapse these into one pool. A single pool mixes tactical and strategic meaning and makes duplicate-handling ambiguous.
     - Preferred v2 representation:
       - Maintain a small fixed-size reveal ledger for public reveal-to-hidden-zone events (for example, "revealed from deck, then added to hand").
       - Each ledger item should carry at least:
         - `card_id`
         - `reveal_source` (`deck_search`, `topdeck_look`, `reveal_from_hand`, etc.)
         - `destination_hidden_zone` (`hand`, `top_of_deck`, other hidden bucket)
         - `turns_since_reveal`
         - `same_card_publicly_seen_after_reveal_count`
         - `confidence_still_hidden`
         - `event_valid`
       - Encode the ledger as a multiset/event-memory block:
         - per-item card embedding + metadata MLP
         - pool with `sum` or `sum + max + count`
         - avoid `max` alone because it loses multiplicity
         - avoid `avg` alone because it blurs "one important reveal" vs. "many copies revealed"
     - When the opponent later plays/discards/reveals the same card name:
       - Keep the reveal in `reveal_history_memory` for the rest of the game.
       - For `current_hidden_belief`, reduce confidence or mark the event as partially resolved; do not hard-delete by default.
       - Reason: if duplicates exist, we usually cannot know whether the public card was the previously revealed copy or another copy.
     - Hard-fact vs uncertain channels:
       - Hard/public counters:
         - `revealed_seen_ever_by_type`
         - `revealed_seen_ever_by_keyword`
         - `revealed_to_hand_count`
         - `revealed_to_hand_recent_count`
       - Uncertain/current-belief counters:
         - `p_still_in_hand[card_or_type]`
         - `known_in_hand_lower_bound[card_or_type]`
         - `hand_belief_entropy`
       - Lower-bound style features are especially useful when duplicate ambiguity exists.
     - Recommended v2 rollout:
       - Step 1: ledger + count/histogram summaries + recency features.
       - Step 2: derive a small probabilistic head for `p_still_in_hand` from public evidence.
       - Step 3: only if needed, move to a richer belief tracker or event-attention module.
     - Alternative approaches:
       - LSTM-only memory:
         - cheapest, but easy to forget long-gap reveals.
       - Full Bayesian/card-count tracker:
         - strongest interpretation of hidden information, but substantially more engineering.
       - Transformer over event tokens:
         - expressive, but likely heavier than needed for first v2 pass.

8. Turn-economy / tempo globals
   - `turn_number`
   - `decision_index_in_turn`
   - `cards_played_this_turn_self`
   - `cards_played_this_turn_opp`
   - `attacks_declared_this_turn_self`
   - `attacks_declared_this_turn_opp`
   - `playable_hand_count_self`
   - `activatable_ability_count_self`
   - `untapped_attackers_self`
   - `untapped_blockers_opp`
   - `priority_holder` (`self | opp`)
   - `passes_in_current_chain`
   - Notes:
     - Priority/pass chain:
       - who can currently act;
       - count of consecutive passes in the active response window (often governs phase advance/stack resolution).

9. Short event-history summaries
   - Per-turn / recent-window aggregates:
     - damage dealt/taken (self/opp)
     - cards drawn/played/discarded (self/opp)
     - removals/revives triggered recently
     - last-N event counts by event type (small fixed histogram)
   - Notes:
     - Helps credit assignment and tactical context without full long sequence replay in observation.

10. Optional expanded previous-action history
   - After v1 previous action tuple, consider:
     - `prev2_*` / last-N self action stats.
     - last-N opponent action stats or opponent action tokens.
     - keep self vs opponent history in separate channels unless each token includes an explicit actor marker.
   - Notes:
     - Keep small unless clear gain.

11. Optional policy-side memory enhancements (if needed later)
   - Keep observation lean and rely more on recurrent hidden state for long horizon context.
   - Add explicit memory channels only if training indicates persistent blind spots.

## 2) Highest ROI Implementation Order (First -> Last)

1. Previous action tuple (self)
   - Why first:
     - Minimal implementation and model-cost.
     - Typically immediate behavioral lift.

2. Active combat / response-window attacker context
   - Why second:
     - Extremely high tactical value in defender response windows.
     - Public information, cheap to expose, and directly answers "who is attacking now?"

3. Opponent previous action tuple / short recent action history
   - Why third:
     - Natural extension of the existing self previous-action block.
     - Gives the defender immediate context for what the opponent just did, especially attack declarations.

4. Ability dynamic readiness state
   - Why fourth:
     - High tactical value: whether a card/ability is actually usable now or soon.
     - Directly impacts action legality and planning.

5. Effect embedding payload with duration + source metadata
   - Why fifth:
     - Scales better than adding ad-hoc booleans forever.
     - Captures rich status semantics with controlled dimensional growth.

6. Turn-economy / tempo globals
   - Why sixth:
     - Strong signal for phase pacing and local tactical tempo.
     - Cheap relative to complex relational/belief systems.

7. Hidden-information belief features (v1 compact)
   - Why seventh:
     - High strategic upside in partially observable play.
     - More engineering complexity; add after stable baseline features.

8. Board relation matrices / threat summaries
   - Why eighth:
     - Powerful, but potentially expensive and easy to overgrow.
     - Add once earlier gains plateau.

9. Short event-history summaries
   - Why ninth:
     - Useful context compression; moderate complexity.
     - Can overlap with what LSTM already captures.

10. Optional expanded history (self/opp last-N actions) and other long-tail extras
   - Why last:
     - Diminishing returns unless diagnostics identify specific deficits.

## 3) Practical Rollout Notes

1. Start with low-risk small-dimensional additions.
2. Keep hard factual channels separate from uncertain/inferred channels.
3. For uncertain channels, always expose uncertainty (`entropy`, `confidence`).
4. In combat/response windows, expose public attacker/target identity directly instead of forcing the model to reconstruct it from hidden state or long action history.
5. If opponent action history is added, start with a single last-opponent-action parity block and only expand to last-`N` if diagnostics still show response-phase ambiguity.
6. Prefer structured set/relation encoders before trying CNNs for this board format.
7. Add instrumentation per feature block so ablations can measure ROI cleanly.
8. For public reveals, preserve both:
   - a short-horizon "may still be in hand" representation;
   - a long-horizon "this deck has shown this card/package" representation.
9. If a pooled set is used for reveal memory, include explicit count or sum-based aggregation so duplicate reveals are not silently erased.
