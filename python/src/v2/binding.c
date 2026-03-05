#include "tcg.h"

#define Env CAzukiTCG
#include "env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
  env->seed = unpack(kwargs, "seed");
  init(env);
  return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "p0_episode_return", log->p0_episode_return);
    assign_to_dict(dict, "p1_episode_return", log->p1_episode_return);
    assign_to_dict(dict, "p0_winrate", log->p0_winrate);
    assign_to_dict(dict, "p1_winrate", log->p1_winrate);
    assign_to_dict(dict, "p0_start_rate", log->p0_start_rate);
    assign_to_dict(dict, "p1_start_rate", log->p1_start_rate);
    assign_to_dict(dict, "draw_rate", log->draw_rate);
    assign_to_dict(dict, "timeout_truncation_rate", log->timeout_truncation_rate);
    assign_to_dict(dict, "auto_tick_truncation_rate", log->auto_tick_truncation_rate);
    assign_to_dict(dict, "gameover_terminal_rate", log->gameover_terminal_rate);
    assign_to_dict(dict, "winner_terminal_rate", log->winner_terminal_rate);
    assign_to_dict(dict, "curriculum_episode_cap", log->curriculum_episode_cap);
    assign_to_dict(dict, "reward_shaping_scale", log->reward_shaping_scale);
    assign_to_dict(dict, "completed_episodes", log->completed_episodes);
    assign_to_dict(dict, "p0_noop_selected_rate", log->p0_noop_selected_rate);
    assign_to_dict(dict, "p1_noop_selected_rate", log->p1_noop_selected_rate);
    assign_to_dict(dict, "p0_attack_selected_rate", log->p0_attack_selected_rate);
    assign_to_dict(dict, "p1_attack_selected_rate", log->p1_attack_selected_rate);
    assign_to_dict(dict, "p0_attach_weapon_from_hand_selected_rate", log->p0_attach_weapon_from_hand_selected_rate);
    assign_to_dict(dict, "p1_attach_weapon_from_hand_selected_rate", log->p1_attach_weapon_from_hand_selected_rate);
    assign_to_dict(dict, "p0_play_spell_from_hand_selected_rate", log->p0_play_spell_from_hand_selected_rate);
    assign_to_dict(dict, "p1_play_spell_from_hand_selected_rate", log->p1_play_spell_from_hand_selected_rate);
    assign_to_dict(dict, "p0_activate_garden_or_leader_ability_selected_rate", log->p0_activate_garden_or_leader_ability_selected_rate);
    assign_to_dict(dict, "p1_activate_garden_or_leader_ability_selected_rate", log->p1_activate_garden_or_leader_ability_selected_rate);
    assign_to_dict(dict, "p0_activate_alley_ability_selected_rate", log->p0_activate_alley_ability_selected_rate);
    assign_to_dict(dict, "p1_activate_alley_ability_selected_rate", log->p1_activate_alley_ability_selected_rate);
    assign_to_dict(dict, "p0_gate_portal_selected_rate", log->p0_gate_portal_selected_rate);
    assign_to_dict(dict, "p1_gate_portal_selected_rate", log->p1_gate_portal_selected_rate);
    assign_to_dict(dict, "p0_play_entity_to_alley_selected_rate", log->p0_play_entity_to_alley_selected_rate);
    assign_to_dict(dict, "p1_play_entity_to_alley_selected_rate", log->p1_play_entity_to_alley_selected_rate);
    assign_to_dict(dict, "p0_play_entity_to_garden_selected_rate", log->p0_play_entity_to_garden_selected_rate);
    assign_to_dict(dict, "p1_play_entity_to_garden_selected_rate", log->p1_play_entity_to_garden_selected_rate);
    assign_to_dict(dict, "p0_play_selected_rate", log->p0_play_selected_rate);
    assign_to_dict(dict, "p1_play_selected_rate", log->p1_play_selected_rate);
    assign_to_dict(dict, "p0_ability_selected_rate", log->p0_ability_selected_rate);
    assign_to_dict(dict, "p1_ability_selected_rate", log->p1_ability_selected_rate);
    assign_to_dict(dict, "p0_target_selected_rate", log->p0_target_selected_rate);
    assign_to_dict(dict, "p1_target_selected_rate", log->p1_target_selected_rate);
    assign_to_dict(dict, "p0_avg_leader_health", log->p0_avg_leader_health);
    assign_to_dict(dict, "p1_avg_leader_health", log->p1_avg_leader_health);
    assign_to_dict(dict, "n", log->n);
    return 0;
}
