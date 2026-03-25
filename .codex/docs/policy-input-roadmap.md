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

2. Effect system upgrade: effect embeddings instead of only fixed booleans
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

3. Ability dynamic readiness state
   - For each relevant ability/card:
     - `can_use_now`
     - `turns_until_usable`
     - `cooldown_remaining`
     - `cost_shortfall` (how far from payable)
     - `legal_target_count`
   - Notes:
     - Distinct from static ability metadata (`ability_timing`, `has_ability`, etc.).

4. Board relation/topology features (non-CNN default)
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

5. Hidden-information belief state (no information leakage)
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

6. Turn-economy / tempo globals
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

7. Short event-history summaries
   - Per-turn / recent-window aggregates:
     - damage dealt/taken (self/opp)
     - cards drawn/played/discarded (self/opp)
     - removals/revives triggered recently
     - last-N event counts by event type (small fixed histogram)
   - Notes:
     - Helps credit assignment and tactical context without full long sequence replay in observation.

8. Optional expanded previous-action history
   - After v1 previous action tuple, consider:
     - `prev2_*` or last-N compressed action stats.
   - Notes:
     - Keep small unless clear gain.

9. Optional policy-side memory enhancements (if needed later)
   - Keep observation lean and rely more on recurrent hidden state for long horizon context.
   - Add explicit memory channels only if training indicates persistent blind spots.

## 2) Highest ROI Implementation Order (First -> Last)

1. Previous action tuple (self)
   - Why first:
     - Minimal implementation and model-cost.
     - Typically immediate behavioral lift.

2. Ability dynamic readiness state
   - Why second:
     - High tactical value: whether a card/ability is actually usable now or soon.
     - Directly impacts action legality and planning.

3. Effect embedding payload with duration + source metadata
   - Why third:
     - Scales better than adding ad-hoc booleans forever.
     - Captures rich status semantics with controlled dimensional growth.

4. Turn-economy / tempo globals
   - Why fourth:
     - Strong signal for phase pacing and local tactical tempo.
     - Cheap relative to complex relational/belief systems.

5. Hidden-information belief features (v1 compact)
   - Why fifth:
     - High strategic upside in partially observable play.
     - More engineering complexity; add after stable baseline features.

6. Board relation matrices / threat summaries
   - Why sixth:
     - Powerful, but potentially expensive and easy to overgrow.
     - Add once earlier gains plateau.

7. Short event-history summaries
   - Why seventh:
     - Useful context compression; moderate complexity.
     - Can overlap with what LSTM already captures.

8. Optional expanded history (prev2/last-N actions) and other long-tail extras
   - Why last:
     - Diminishing returns unless diagnostics identify specific deficits.

## 3) Practical Rollout Notes

1. Start with low-risk small-dimensional additions.
2. Keep hard factual channels separate from uncertain/inferred channels.
3. For uncertain channels, always expose uncertainty (`entropy`, `confidence`).
4. Prefer structured set/relation encoders before trying CNNs for this board format.
5. Add instrumentation per feature block so ablations can measure ROI cleanly.
6. For public reveals, preserve both:
   - a short-horizon "may still be in hand" representation;
   - a long-horizon "this deck has shown this card/package" representation.
7. If a pooled set is used for reveal memory, include explicit count or sum-based aggregation so duplicate reveals are not silently erased.
