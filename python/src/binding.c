#include "tcg.h"

#include <Python.h>

static PyObject* env_reset_with_decks(PyObject* self, PyObject* args);
static PyObject* vec_drain_deck_records(PyObject* self, PyObject* args);
static PyObject* vec_reset_evaluation_games(PyObject* self, PyObject* args);
static PyObject* vec_active_players(PyObject* self, PyObject* args);
static PyObject* vec_force_evaluation_truncations(PyObject* self, PyObject* args);
static PyObject* obs_struct_sizes(PyObject* self, PyObject* args);

#define MY_METHODS \
  {"env_reset_with_decks", env_reset_with_decks, METH_VARARGS, "Reset the environment with two explicit deck specs"}, \
  {"vec_drain_deck_records", vec_drain_deck_records, METH_VARARGS, "Drain per-episode drafted-deck records from a deck-building vec"}, \
  {"vec_reset_evaluation_games", vec_reset_evaluation_games, METH_VARARGS, "Reset selected native evaluation games with explicit seeds, gates, and optional reference seats"}, \
  {"vec_active_players", vec_active_players, METH_VARARGS, "Return the active player for each native vector environment"}, \
  {"vec_force_evaluation_truncations", vec_force_evaluation_truncations, METH_VARARGS, "Force selected native evaluation games to truncate"}, \
  {"obs_struct_sizes", obs_struct_sizes, METH_NOARGS, "Return (battle, deckbuild) packed observation struct sizes"}

#define Env CAzukiTCG
#define MY_GET
#include "env_binding.h"

static int lookup_card_def_id(const char *card_code, CardDefId *out_card_id) {
  size_t lookup_count = 0;
  const CardDefLookupEntry *lookup = azk_card_def_lookup_table(&lookup_count);
  for (size_t index = 0; index < lookup_count; ++index) {
    if (strcmp(lookup[index].card_id, card_code) == 0) {
      *out_card_id = (CardDefId)index;
      return 0;
    }
  }

  PyErr_Format(PyExc_ValueError, "Unknown training deck card code '%s'", card_code);
  return -1;
}

static int parse_card_quantity(PyObject *value, const char *card_code) {
  if (!PyLong_Check(value)) {
    PyErr_Format(PyExc_TypeError, "Quantity for card '%s' must be an integer", card_code);
    return -1;
  }

  long quantity = PyLong_AsLong(value);
  if (PyErr_Occurred()) {
    return -1;
  }
  if (quantity <= 0 || quantity > UINT8_MAX) {
    PyErr_Format(
        PyExc_ValueError,
        "Quantity for card '%s' must be between 1 and %u (got %ld)",
        card_code,
        (unsigned int)UINT8_MAX,
        quantity);
    return -1;
  }
  return (int)quantity;
}

static int parse_deck_spec(
    PyObject *deck_obj,
    const char *deck_label,
    CardInfo **out_cards,
    size_t *out_card_count) {
  *out_cards = NULL;
  *out_card_count = 0;
  if (!PySequence_Check(deck_obj)) {
    PyErr_Format(
        PyExc_TypeError,
        "%s must be a sequence of (card_id, quantity) pairs",
        deck_label);
    return -1;
  }

  const Py_ssize_t card_count = PySequence_Size(deck_obj);
  if (card_count < 0) {
    return -1;
  }
  if (card_count <= 0) {
    PyErr_Format(PyExc_ValueError, "%s must contain at least one card entry", deck_label);
    return -1;
  }

  CardInfo *cards = calloc((size_t)card_count, sizeof(CardInfo));
  if (cards == NULL) {
    PyErr_Format(PyExc_MemoryError, "Failed to allocate %s card array", deck_label);
    return -1;
  }

  int total_cards = 0;
  for (Py_ssize_t card_index = 0; card_index < card_count; ++card_index) {
    PyObject *card_entry_obj = PySequence_GetItem(deck_obj, card_index);
    if (card_entry_obj == NULL) {
      free(cards);
      return -1;
    }

    if (!PySequence_Check(card_entry_obj)) {
      Py_DECREF(card_entry_obj);
      free(cards);
      PyErr_Format(
          PyExc_TypeError,
          "%s[%zd] must be a (card_id, quantity) pair",
          deck_label,
          card_index);
      return -1;
    }

    const Py_ssize_t card_entry_size = PySequence_Size(card_entry_obj);
    if (card_entry_size < 0) {
      Py_DECREF(card_entry_obj);
      free(cards);
      return -1;
    }
    if (card_entry_size != 2) {
      Py_DECREF(card_entry_obj);
      free(cards);
      PyErr_Format(
          PyExc_ValueError,
          "%s[%zd] must contain exactly 2 items: (card_id, quantity)",
          deck_label,
          card_index);
      return -1;
    }

    PyObject *card_code_obj = PySequence_GetItem(card_entry_obj, 0);
    PyObject *quantity_obj = PySequence_GetItem(card_entry_obj, 1);
    if (card_code_obj == NULL || quantity_obj == NULL) {
      Py_XDECREF(card_code_obj);
      Py_XDECREF(quantity_obj);
      Py_DECREF(card_entry_obj);
      free(cards);
      return -1;
    }
    if (!PyUnicode_Check(card_code_obj)) {
      Py_DECREF(card_code_obj);
      Py_DECREF(quantity_obj);
      Py_DECREF(card_entry_obj);
      free(cards);
      PyErr_Format(PyExc_TypeError, "%s[%zd] card_id must be a string", deck_label, card_index);
      return -1;
    }

    const char *card_code = PyUnicode_AsUTF8(card_code_obj);
    if (card_code == NULL) {
      Py_DECREF(card_code_obj);
      Py_DECREF(quantity_obj);
      Py_DECREF(card_entry_obj);
      free(cards);
      return -1;
    }

    CardDefId card_id;
    if (lookup_card_def_id(card_code, &card_id) != 0) {
      Py_DECREF(card_code_obj);
      Py_DECREF(quantity_obj);
      Py_DECREF(card_entry_obj);
      free(cards);
      return -1;
    }

    const int quantity = parse_card_quantity(quantity_obj, card_code);
    if (quantity < 0) {
      Py_DECREF(card_code_obj);
      Py_DECREF(quantity_obj);
      Py_DECREF(card_entry_obj);
      free(cards);
      return -1;
    }

    cards[card_index].card_id = card_id;
    cards[card_index].card_count = quantity;
    total_cards += quantity;
    Py_DECREF(card_code_obj);
    Py_DECREF(quantity_obj);
    Py_DECREF(card_entry_obj);
  }

  const int expected_total_cards = REQUIRED_DECK_SIZE + REQUIRED_LEADER_SIZE +
                                   REQUIRED_GATE_SIZE + REQUIRED_IKZ_PILE_SIZE;
  if (total_cards != expected_total_cards) {
    free(cards);
    PyErr_Format(
        PyExc_ValueError,
        "%s resolves to %d total cards; expected %d",
        deck_label,
        total_cards,
        expected_total_cards);
    return -1;
  }

  *out_cards = cards;
  *out_card_count = (size_t)card_count;
  return 0;
}

static int load_training_deck_pool(Env *env, PyObject *deck_pool_obj) {
  if (!PySequence_Check(deck_pool_obj)) {
    PyErr_SetString(
        PyExc_TypeError,
        "deck_pool must be a sequence of decks; each deck must be a sequence of (card_id, quantity) pairs");
    return -1;
  }

  const Py_ssize_t deck_pool_count = PySequence_Size(deck_pool_obj);
  if (deck_pool_count < 0) {
    return -1;
  }
  if (deck_pool_count <= 0) {
    PyErr_SetString(PyExc_ValueError, "deck_pool must contain at least one deck");
    return -1;
  }

  free_training_deck_pool(env);
  env->deck_pool = calloc((size_t)deck_pool_count, sizeof(TrainingDeckSpec));
  if (env->deck_pool == NULL) {
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate training deck pool");
    return -1;
  }
  env->deck_pool_count = (size_t)deck_pool_count;
  reset_current_deck_indices(env);

  for (Py_ssize_t deck_index = 0; deck_index < deck_pool_count; ++deck_index) {
    PyObject *deck_obj = PySequence_GetItem(deck_pool_obj, deck_index);
    if (deck_obj == NULL) {
      free_training_deck_pool(env);
      return -1;
    }

    if (!PySequence_Check(deck_obj)) {
      Py_DECREF(deck_obj);
      free_training_deck_pool(env);
      PyErr_Format(
          PyExc_TypeError,
          "deck_pool[%zd] must be a sequence of (card_id, quantity) pairs",
          deck_index);
      return -1;
    }

    CardInfo *cards = NULL;
    size_t card_count = 0;
    char deck_label[64];
    snprintf(deck_label, sizeof(deck_label), "deck_pool[%zd]", deck_index);
    if (parse_deck_spec(deck_obj, deck_label, &cards, &card_count) != 0) {
      Py_DECREF(deck_obj);
      free_training_deck_pool(env);
      return -1;
    }
    Py_DECREF(deck_obj);

    env->deck_pool[deck_index].cards = cards;
    env->deck_pool[deck_index].card_count = card_count;
  }

  return 0;
}

static PyObject* my_get(PyObject* dict, Env* env) {
  PyObject *deck_indices = PyList_New(MAX_PLAYERS_PER_MATCH);
  if (deck_indices == NULL) {
    return NULL;
  }

  for (int player_index = 0; player_index < MAX_PLAYERS_PER_MATCH; ++player_index) {
    PyObject *value = PyLong_FromLong(env->current_deck_indices[player_index]);
    if (value == NULL) {
      Py_DECREF(deck_indices);
      return NULL;
    }
    PyList_SET_ITEM(deck_indices, player_index, value);
  }

  PyObject *deck_pool_count = PyLong_FromSize_t(env->deck_pool_count);
  if (deck_pool_count == NULL) {
    Py_DECREF(deck_indices);
    return NULL;
  }

  if (PyDict_SetItemString(dict, "current_deck_indices", deck_indices) < 0 ||
      PyDict_SetItemString(dict, "deck_pool_count", deck_pool_count) < 0) {
    Py_DECREF(deck_indices);
    Py_DECREF(deck_pool_count);
    return NULL;
  }

  Py_DECREF(deck_indices);
  Py_DECREF(deck_pool_count);
  return dict;
}

static int load_draft_catalog(PyObject* kwargs);

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
  env->seed = unpack(kwargs, "seed");
  if (PyErr_Occurred()) {
    return -1;
  }

  PyObject *deck_pool_obj = PyDict_GetItemString(kwargs, "deck_pool");
  if (deck_pool_obj != NULL && load_training_deck_pool(env, deck_pool_obj) != 0) {
    return -1;
  }

  PyObject *deck_building_obj = PyDict_GetItemString(kwargs, "deck_building");
  if (deck_building_obj != NULL && PyObject_IsTrue(deck_building_obj)) {
    if (load_draft_catalog(kwargs) != 0) {
      free_training_deck_pool(env);
      return -1;
    }
    env->deck_building = true;
    env->evaluation_forced_gate[0] = -1;
    env->evaluation_forced_gate[1] = -1;
    env->evaluation_forced_leader[0] = -1;
    env->evaluation_forced_leader[1] = -1;
    env->evaluation_reference_seat = -1;
    env->evaluation_reference_deck_index = -1;
    PyObject* pause_obj =
        PyDict_GetItemString(kwargs, "evaluation_pause_on_done");
    if (pause_obj != NULL && PyObject_IsTrue(pause_obj)) {
      env->evaluation_pause_on_done = true;
    }
    PyObject* uniform_assignment_obj =
        PyDict_GetItemString(kwargs, "draft_uniform_assignment");
    if (uniform_assignment_obj != NULL && PyObject_IsTrue(uniform_assignment_obj)) {
      env->draft_uniform_assignment = true;
    }
    PyObject* sibling_prob_obj =
        PyDict_GetItemString(kwargs, "draft_same_element_matchup_prob");
    if (sibling_prob_obj != NULL) {
      const double prob = PyFloat_AsDouble(sibling_prob_obj);
      if (PyErr_Occurred()) {
        free_training_deck_pool(env);
        return -1;
      }
      if (prob < 0.0 || prob > 1.0) {
        PyErr_SetString(PyExc_ValueError,
                        "draft_same_element_matchup_prob must be in [0, 1]");
        free_training_deck_pool(env);
        return -1;
      }
      env->draft_same_element_matchup_prob = (float)prob;
    }
    PyObject* xgate_obj =
        PyDict_GetItemString(kwargs, "draft_cross_gate_replay_prob");
    if (xgate_obj != NULL) {
      const double prob = PyFloat_AsDouble(xgate_obj);
      if (PyErr_Occurred()) {
        free_training_deck_pool(env);
        return -1;
      }
      if (prob < 0.0 || prob > 1.0) {
        PyErr_SetString(PyExc_ValueError,
                        "draft_cross_gate_replay_prob must be in [0, 1]");
        free_training_deck_pool(env);
        return -1;
      }
      env->draft_cross_gate_replay_prob = (float)prob;
    }
    PyObject* priv_decks_obj =
        PyDict_GetItemString(kwargs, "deck_building_privileged_decks");
    if (priv_decks_obj != NULL && PyObject_IsTrue(priv_decks_obj)) {
      env->deck_building_privileged_decks = true;
    }
  }

  init(env);
  if (!env->deck_building && env->engine == NULL) {
    const char *error_message = azk_engine_get_last_error();
    PyErr_SetString(
        PyExc_RuntimeError,
        error_message != NULL ? error_message : "Failed to initialize Azuki engine");
    free_training_deck_pool(env);
    return -1;
  }
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
    assign_to_dict(dict, "zero_legal_action_truncation_rate", log->zero_legal_action_truncation_rate);
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
    assign_to_dict(dict, "p0_entity_damage_dealt", log->p0_entity_damage_dealt);
    assign_to_dict(dict, "p1_entity_damage_dealt", log->p1_entity_damage_dealt);
    assign_to_dict(dict, "p0_entity_damage_taken", log->p0_entity_damage_taken);
    assign_to_dict(dict, "p1_entity_damage_taken", log->p1_entity_damage_taken);
    assign_to_dict(dict, "p0_generated_ikz_created", log->p0_generated_ikz_created);
    assign_to_dict(dict, "p1_generated_ikz_created", log->p1_generated_ikz_created);
    assign_to_dict(dict, "p0_generated_ikz_converted", log->p0_generated_ikz_converted);
    assign_to_dict(dict, "p1_generated_ikz_converted", log->p1_generated_ikz_converted);
    assign_to_dict(dict, "p0_generated_ikz_conversion_rate", log->p0_generated_ikz_conversion_rate);
    assign_to_dict(dict, "p1_generated_ikz_conversion_rate", log->p1_generated_ikz_conversion_rate);
    assign_to_dict(dict, "p0_temporary_charge_realized", log->p0_temporary_charge_realized);
    assign_to_dict(dict, "p1_temporary_charge_realized", log->p1_temporary_charge_realized);
    assign_to_dict(dict, "p0_temporary_attack_damage_realized", log->p0_temporary_attack_damage_realized);
    assign_to_dict(dict, "p1_temporary_attack_damage_realized", log->p1_temporary_attack_damage_realized);
    assign_to_dict(dict, "p0_contextual_response_reserve_opportunities", log->p0_contextual_response_reserve_opportunities);
    assign_to_dict(dict, "p1_contextual_response_reserve_opportunities", log->p1_contextual_response_reserve_opportunities);
    assign_to_dict(dict, "n", log->n);
    return 0;
}

static PyObject* env_reset_with_decks(PyObject* self, PyObject* args) {
  (void)self;
  if (PyTuple_Size(args) != 4) {
    PyErr_SetString(PyExc_TypeError, "env_reset_with_decks requires 4 arguments: env_handle, seed, player0_deck, player1_deck");
    return NULL;
  }

  Env* env = unpack_env(args);
  if (env == NULL) {
    return NULL;
  }

  PyObject* seed_arg = PyTuple_GetItem(args, 1);
  if (!PyObject_TypeCheck(seed_arg, &PyLong_Type)) {
    PyErr_SetString(PyExc_TypeError, "seed must be an integer");
    return NULL;
  }
  env->seed = PyLong_AsLong(seed_arg);
  if (PyErr_Occurred()) {
    return NULL;
  }
  env->starter_rng_state = starter_seed_from_env_seed(env->seed);
  env->deck_rng_state = deck_seed_from_env_seed(env->seed);

  CardInfo *player0_cards = NULL;
  CardInfo *player1_cards = NULL;
  size_t player0_card_count = 0;
  size_t player1_card_count = 0;
  PyObject *player0_deck = PyTuple_GetItem(args, 2);
  PyObject *player1_deck = PyTuple_GetItem(args, 3);
  if (parse_deck_spec(player0_deck, "player0_deck", &player0_cards, &player0_card_count) != 0) {
    return NULL;
  }
  if (parse_deck_spec(player1_deck, "player1_deck", &player1_cards, &player1_card_count) != 0) {
    free(player0_cards);
    return NULL;
  }

  c_reset_with_decks(
      env,
      player0_cards,
      player0_card_count,
      player1_cards,
      player1_card_count);
  free(player0_cards);
  free(player1_cards);
  Py_RETURN_NONE;
}

// ---- Deck-building draft catalog + export drain -----------------------------

static int parse_int16_list(PyObject* kwargs, const char* key, int16_t* out,
                            int capacity, int* out_count) {
  PyObject* obj = PyDict_GetItemString(kwargs, key);
  if (obj == NULL) {
    PyErr_Format(PyExc_ValueError, "deck_building requires kwarg '%s'", key);
    return -1;
  }
  PyObject* seq = PySequence_Fast(obj, "draft catalog entries must be sequences");
  if (seq == NULL) {
    return -1;
  }
  const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
  if (n > capacity) {
    Py_DECREF(seq);
    PyErr_Format(PyExc_ValueError, "'%s' has %zd entries, capacity %d", key, n,
                 capacity);
    return -1;
  }
  for (Py_ssize_t i = 0; i < n; ++i) {
    const long value = PyLong_AsLong(PySequence_Fast_GET_ITEM(seq, i));
    if (PyErr_Occurred()) {
      Py_DECREF(seq);
      return -1;
    }
    out[i] = (int16_t)value;
  }
  Py_DECREF(seq);
  *out_count = (int)n;
  return 0;
}

static int parse_int_list(PyObject* kwargs, const char* key, int* out,
                          int capacity, int* out_count) {
  PyObject* obj = PyDict_GetItemString(kwargs, key);
  if (obj == NULL) {
    PyErr_Format(PyExc_ValueError, "deck_building requires kwarg '%s'", key);
    return -1;
  }
  PyObject* seq = PySequence_Fast(obj, "draft catalog entries must be sequences");
  if (seq == NULL) {
    return -1;
  }
  const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
  if (n > capacity) {
    Py_DECREF(seq);
    PyErr_Format(PyExc_ValueError, "'%s' has %zd entries, capacity %d", key, n,
                 capacity);
    return -1;
  }
  for (Py_ssize_t i = 0; i < n; ++i) {
    const long value = PyLong_AsLong(PySequence_Fast_GET_ITEM(seq, i));
    if (PyErr_Occurred()) {
      Py_DECREF(seq);
      return -1;
    }
    out[i] = (int)value;
  }
  Py_DECREF(seq);
  *out_count = (int)n;
  return 0;
}

static int load_draft_catalog(PyObject* kwargs) {
  if (g_draft_catalog.loaded) {
    return 0;
  }
  AzkDraftCatalog cat = {0};
  int count = 0;
  if (parse_int16_list(kwargs, "draft_gate_def_ids", cat.gate_def_ids,
                       AZK_DRAFT_MAX_GATES, &cat.gate_count) != 0 ||
      parse_int16_list(kwargs, "draft_gate_population", cat.gate_population,
                       AZK_DRAFT_MAX_POPULATION, &cat.population_count) != 0 ||
      parse_int16_list(kwargs, "draft_leader_flat", cat.leader_flat,
                       AZK_DRAFT_MAX_GATES * AZK_DRAFT_MAX_LEADERS, &count) != 0 ||
      parse_int_list(kwargs, "draft_leader_offsets", cat.leader_offsets,
                     AZK_DRAFT_MAX_GATES + 1, &count) != 0 ||
      parse_int16_list(kwargs, "draft_main_flat", cat.main_flat,
                       AZK_DRAFT_MAX_GATES * AZK_DRAFT_MAX_CANDIDATES, &count) != 0 ||
      parse_int_list(kwargs, "draft_main_offsets", cat.main_offsets,
                     AZK_DRAFT_MAX_GATES + 1, &count) != 0) {
    return -1;
  }
  if (count != cat.gate_count + 1) {
    PyErr_SetString(PyExc_ValueError,
                    "draft offsets must have gate_count+1 entries");
    return -1;
  }
  PyObject* ikz_obj = PyDict_GetItemString(kwargs, "draft_ikz_def_id");
  if (ikz_obj == NULL) {
    PyErr_SetString(PyExc_ValueError, "deck_building requires draft_ikz_def_id");
    return -1;
  }
  cat.ikz_def_id = (int16_t)PyLong_AsLong(ikz_obj);
  if (PyErr_Occurred()) {
    return -1;
  }
  for (int i = 0; i < AZK_DRAFT_MAX_GATES; ++i) {
    cat.gate_sibling_def_ids[i] = -1;
  }
  if (PyDict_GetItemString(kwargs, "draft_gate_sibling_def_ids") != NULL) {
    int sibling_count = 0;
    if (parse_int16_list(kwargs, "draft_gate_sibling_def_ids",
                         cat.gate_sibling_def_ids, AZK_DRAFT_MAX_GATES,
                         &sibling_count) != 0) {
      return -1;
    }
    if (sibling_count != cat.gate_count) {
      PyErr_SetString(PyExc_ValueError,
                      "draft_gate_sibling_def_ids must match gate count");
      return -1;
    }
    for (int i = 0; i < cat.gate_count; ++i) {
      const int16_t sibling = cat.gate_sibling_def_ids[i];
      if (sibling < 0) {
        continue;
      }
      bool found = false;
      for (int j = 0; j < cat.gate_count; ++j) {
        if (cat.gate_def_ids[j] == sibling) {
          found = true;
          break;
        }
      }
      if (!found) {
        PyErr_Format(PyExc_ValueError,
                     "draft_gate_sibling_def_ids[%d]=%d not a catalog gate", i,
                     (int)sibling);
        return -1;
      }
    }
  }
  if (cat.gate_count <= 0 || cat.population_count <= 0) {
    PyErr_SetString(PyExc_ValueError, "draft catalog must be non-empty");
    return -1;
  }
  // Per-gate main lists must fit the per-player copy-count arrays.
  for (int g = 0; g < cat.gate_count; ++g) {
    const int len = cat.main_offsets[g + 1] - cat.main_offsets[g];
    if (len <= 0 || len > AZK_DRAFT_MAX_CANDIDATES) {
      PyErr_Format(PyExc_ValueError,
                   "gate %d main candidate list length %d out of range", g, len);
      return -1;
    }
  }
  g_draft_catalog = cat;
  g_draft_catalog.loaded = true;
  return 0;
}

static PyObject* vec_drain_deck_records(PyObject* self, PyObject* args) {
  (void)self;
  VecEnv* vec = unpack_vecenv(args);
  if (vec == NULL) {
    return NULL;
  }
  PyObject* records = PyList_New(0);
  if (records == NULL) {
    return NULL;
  }
  for (int i = 0; i < vec->num_envs; ++i) {
    Env* env = vec->envs[i];
    if (!env->deck_record_valid) {
      continue;
    }
    env->deck_record_valid = false;
    PyObject* players = PyList_New(MAX_PLAYERS_PER_MATCH);
    if (players == NULL) {
      Py_DECREF(records);
      return NULL;
    }
    for (int p = 0; p < MAX_PLAYERS_PER_MATCH; ++p) {
      PyObject* main_list = PyList_New(REQUIRED_DECK_SIZE);
      if (main_list == NULL) {
        Py_DECREF(players);
        Py_DECREF(records);
        return NULL;
      }
      for (int c = 0; c < REQUIRED_DECK_SIZE; ++c) {
        PyList_SET_ITEM(main_list, c,
                        PyLong_FromLong(env->deck_record_main[p][c]));
      }
      PyObject* player = Py_BuildValue(
          "{s:i,s:i,s:N,s:f,s:f,s:f,s:f,s:f,s:f,s:f,s:f,s:f}",
          "gate", (int)env->deck_record_gate[p],
          "leader", (int)env->deck_record_leader[p],
          "main", main_list,
          "win", env->deck_record_win[p],
          "attack_rate", env->deck_record_behavior[p][0],
          "spell_rate", env->deck_record_behavior[p][1],
          "weapon_rate", env->deck_record_behavior[p][2],
          "portal_rate", env->deck_record_behavior[p][3],
          "play_entity_rate", env->deck_record_behavior[p][4],
          "noop_rate", env->deck_record_behavior[p][5],
          "ability_rate",
          env->deck_record_behavior[p][6] + env->deck_record_behavior[p][7],
          "leader_health", env->deck_record_leader_health[p]);
      if (player == NULL) {
        Py_DECREF(players);
        Py_DECREF(records);
        return NULL;
      }
      assign_to_dict(player, "garden_or_leader_ability_rate",
                     env->deck_record_behavior[p][6]);
      assign_to_dict(player, "alley_ability_rate",
                     env->deck_record_behavior[p][7]);
      assign_to_dict(player, "play_entity_to_garden_rate",
                     env->deck_record_behavior[p][8]);
      assign_to_dict(player, "play_entity_to_alley_rate",
                     env->deck_record_behavior[p][9]);
      assign_to_dict(player, "target_rate", env->deck_record_behavior[p][10]);
      assign_to_dict(player, "contextual_response_opportunities",
                     env->deck_record_behavior[p][11]);
      assign_to_dict(player, "temporary_charge_realized",
                     env->deck_record_behavior[p][12]);
      assign_to_dict(player, "temporary_attack_damage_realized",
                     env->deck_record_behavior[p][13]);
      assign_to_dict(player, "generated_ikz_created",
                     env->deck_record_behavior[p][14]);
      assign_to_dict(player, "generated_ikz_converted",
                     env->deck_record_behavior[p][15]);
      assign_to_dict(player, "entity_damage_dealt",
                     env->deck_record_behavior[p][16]);
      assign_to_dict(player, "entity_damage_taken",
                     env->deck_record_behavior[p][17]);
      PyList_SET_ITEM(players, p, player);
    }
    PyObject* record = Py_BuildValue(
        "{s:i,s:k,s:f,s:i,s:i,s:i,s:i,s:N}",
        "env_index", i,
        "seed", (unsigned long)env->deck_record_seed,
        "episode_length", env->deck_record_episode_length,
        "ref_seat", (int)env->deck_record_ref_seat,
        "ref_deck_index", (int)env->deck_record_ref_deck_index,
        "end_reason", (int)env->deck_record_end_reason,
        "starting_player", (int)env->deck_record_starting_player,
        "players", players);
    if (record == NULL) {
      Py_DECREF(records);
      return NULL;
    }
    if (PyList_Append(records, record) < 0) {
      Py_DECREF(record);
      Py_DECREF(records);
      return NULL;
    }
    Py_DECREF(record);
  }
  return records;
}

static PyObject* int_sequence_fast(PyObject* obj, const char* label) {
  if (obj == NULL) {
    PyErr_Format(PyExc_TypeError, "Missing sequence '%s'", label);
    return NULL;
  }
  return PySequence_Fast(obj, label);
}

static int sequence_long_at(PyObject* sequence, Py_ssize_t index, long* out) {
  *out = PyLong_AsLong(PySequence_Fast_GET_ITEM(sequence, index));
  return PyErr_Occurred() ? -1 : 0;
}

static PyObject* vec_reset_evaluation_games(PyObject* self, PyObject* args) {
  (void)self;
  if (PyTuple_Size(args) != 9) {
    PyErr_SetString(
        PyExc_TypeError,
        "vec_reset_evaluation_games requires handle, indices, seeds, gate0, gate1, leader0, leader1, ref_seats, ref_decks");
    return NULL;
  }
  VecEnv* vec = unpack_vecenv(args);
  if (vec == NULL) {
    return NULL;
  }

  const char* labels[8] = {
      "indices", "seeds", "gate0", "gate1", "leader0", "leader1",
      "ref_seats", "ref_decks"};
  PyObject* sequences[8] = {NULL};
  for (int sequence_index = 0; sequence_index < 8; ++sequence_index) {
    sequences[sequence_index] = int_sequence_fast(
        PyTuple_GetItem(args, sequence_index + 1), labels[sequence_index]);
    if (sequences[sequence_index] == NULL) {
      for (int prior = 0; prior < sequence_index; ++prior) {
        Py_DECREF(sequences[prior]);
      }
      return NULL;
    }
  }
  const Py_ssize_t count = PySequence_Fast_GET_SIZE(sequences[0]);
  for (int sequence_index = 1; sequence_index < 8; ++sequence_index) {
    if (PySequence_Fast_GET_SIZE(sequences[sequence_index]) != count) {
      for (int i = 0; i < 8; ++i) {
        Py_DECREF(sequences[i]);
      }
      PyErr_SetString(PyExc_ValueError, "Evaluation reset sequences must have equal lengths");
      return NULL;
    }
  }

  for (Py_ssize_t item = 0; item < count; ++item) {
    long index = -1;
    long seed = 0;
    long gate0 = -1;
    long gate1 = -1;
    long leader0 = -1;
    long leader1 = -1;
    long ref_seat = -1;
    long ref_deck = -1;
    if (sequence_long_at(sequences[0], item, &index) != 0 ||
        sequence_long_at(sequences[1], item, &seed) != 0 ||
        sequence_long_at(sequences[2], item, &gate0) != 0 ||
        sequence_long_at(sequences[3], item, &gate1) != 0 ||
        sequence_long_at(sequences[4], item, &leader0) != 0 ||
        sequence_long_at(sequences[5], item, &leader1) != 0 ||
        sequence_long_at(sequences[6], item, &ref_seat) != 0 ||
        sequence_long_at(sequences[7], item, &ref_deck) != 0) {
      for (int i = 0; i < 8; ++i) {
        Py_DECREF(sequences[i]);
      }
      return NULL;
    }
    if (index < 0 || index >= vec->num_envs) {
      for (int i = 0; i < 8; ++i) {
        Py_DECREF(sequences[i]);
      }
      PyErr_Format(PyExc_IndexError, "Evaluation env index %ld is out of range", index);
      return NULL;
    }
    Env* env = vec->envs[index];
    if (!env->deck_building) {
      for (int i = 0; i < 8; ++i) {
        Py_DECREF(sequences[i]);
      }
      PyErr_SetString(PyExc_ValueError, "Scheduled evaluation requires deck_building mode");
      return NULL;
    }
    const bool random_gates = gate0 < 0 && gate1 < 0;
    if (!random_gates &&
        (gate0 < 0 || gate1 < 0 ||
         draft_gate_slot_for((int16_t)gate0) < 0 ||
         draft_gate_slot_for((int16_t)gate1) < 0)) {
      for (int i = 0; i < 8; ++i) {
        Py_DECREF(sequences[i]);
      }
      PyErr_Format(PyExc_ValueError, "Invalid forced evaluation gates [%ld,%ld]", gate0, gate1);
      return NULL;
    }
    const bool random_leaders = leader0 < 0 && leader1 < 0;
    if (!random_leaders) {
      if (!env->draft_uniform_assignment || random_gates ||
          leader0 < 0 || leader1 < 0 ||
          !draft_leader_valid_for_slot(
              draft_gate_slot_for((int16_t)gate0), (int16_t)leader0) ||
          !draft_leader_valid_for_slot(
              draft_gate_slot_for((int16_t)gate1), (int16_t)leader1)) {
        for (int i = 0; i < 8; ++i) {
          Py_DECREF(sequences[i]);
        }
        PyErr_Format(PyExc_ValueError,
                     "Invalid forced evaluation leaders [%ld,%ld] for gates [%ld,%ld]",
                     leader0, leader1, gate0, gate1);
        return NULL;
      }
    }
    const bool no_reference = ref_seat < 0 && ref_deck < 0;
    if (!no_reference &&
        (ref_seat < 0 || ref_seat >= MAX_PLAYERS_PER_MATCH ||
         ref_deck < 0 || (size_t)ref_deck >= env->deck_pool_count)) {
      for (int i = 0; i < 8; ++i) {
        Py_DECREF(sequences[i]);
      }
      PyErr_Format(
          PyExc_ValueError,
          "Invalid evaluation reference seat/deck [%ld,%ld]", ref_seat, ref_deck);
      return NULL;
    }

    env->evaluation_pause_on_done = true;
    env->evaluation_forced_gates = !random_gates;
    env->evaluation_forced_gate[0] = (int16_t)gate0;
    env->evaluation_forced_gate[1] = (int16_t)gate1;
    env->evaluation_forced_leaders = !random_leaders;
    env->evaluation_forced_leader[0] = (int16_t)leader0;
    env->evaluation_forced_leader[1] = (int16_t)leader1;
    env->evaluation_reference_seat = no_reference ? -1 : (int8_t)ref_seat;
    env->evaluation_reference_deck_index = no_reference ? -1 : (int16_t)ref_deck;
    env->deck_record_valid = false;
    env->seed = (uint32_t)seed;
    env->starter_rng_state = starter_seed_from_env_seed(env->seed);
    env->deck_rng_state = deck_seed_from_env_seed(env->seed);
    env->draft_rng_state = env->seed ^ 0x9E3779B9u;
    c_reset(env);
  }

  for (int i = 0; i < 8; ++i) {
    Py_DECREF(sequences[i]);
  }
  Py_RETURN_NONE;
}

static PyObject* vec_active_players(PyObject* self, PyObject* args) {
  (void)self;
  VecEnv* vec = unpack_vecenv(args);
  if (vec == NULL) {
    return NULL;
  }
  PyObject* active = PyList_New(vec->num_envs);
  if (active == NULL) {
    return NULL;
  }
  for (int i = 0; i < vec->num_envs; ++i) {
    Env* env = vec->envs[i];
    int player = -1;
    const bool done =
        (env->terminals[0] == DONE && env->terminals[1] == DONE) ||
        (env->truncations[0] == DONE && env->truncations[1] == DONE);
    if (!done) {
      player = env->draft_active ? (int)env->draft_active_player
                                 : (int)tcg_active_player_index(env);
    }
    PyList_SET_ITEM(active, i, PyLong_FromLong(player));
  }
  return active;
}

static PyObject* vec_force_evaluation_truncations(PyObject* self, PyObject* args) {
  (void)self;
  if (PyTuple_Size(args) != 2) {
    PyErr_SetString(
        PyExc_TypeError,
        "vec_force_evaluation_truncations requires handle and env indices");
    return NULL;
  }
  VecEnv* vec = unpack_vecenv(args);
  if (vec == NULL) {
    return NULL;
  }
  PyObject* indices = int_sequence_fast(PyTuple_GetItem(args, 1), "indices");
  if (indices == NULL) {
    return NULL;
  }
  const Py_ssize_t count = PySequence_Fast_GET_SIZE(indices);
  for (Py_ssize_t item = 0; item < count; ++item) {
    long index = -1;
    if (sequence_long_at(indices, item, &index) != 0) {
      Py_DECREF(indices);
      return NULL;
    }
    if (index < 0 || index >= vec->num_envs) {
      Py_DECREF(indices);
      PyErr_Format(PyExc_IndexError, "Evaluation env index %ld is out of range", index);
      return NULL;
    }
    Env* env = vec->envs[index];
    const bool done =
        (env->terminals[0] == DONE && env->terminals[1] == DONE) ||
        (env->truncations[0] == DONE && env->truncations[1] == DONE);
    if (done) {
      continue;
    }
    if (env->draft_active || env->engine == NULL) {
      Py_DECREF(indices);
      PyErr_SetString(PyExc_RuntimeError, "Cannot truncate evaluation game during draft");
      return NULL;
    }
    apply_truncation_rewards(env, EP_END_REASON_TIMEOUT_TRUNCATION);
    accumulate_step_rewards(env);
    env->truncations[0] = DONE;
    env->truncations[1] = DONE;
    record_episode_stats(env, EP_END_REASON_TIMEOUT_TRUNCATION);
  }
  Py_DECREF(indices);
  Py_RETURN_NONE;
}

static PyObject* obs_struct_sizes(PyObject* self, PyObject* args) {
  (void)self;
  (void)args;
  return Py_BuildValue("(kk)",
                       (unsigned long)sizeof(TrainingObservationData),
                       (unsigned long)sizeof(TrainingObservationDataDeckBuild));
}
