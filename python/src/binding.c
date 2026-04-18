#include "tcg.h"

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

static int load_training_deck_pool(Env *env, PyObject *deck_pool_obj) {
  PyObject *deck_pool_seq = PySequence_Fast(
      deck_pool_obj,
      "deck_pool must be a sequence of decks; each deck must be a sequence of (card_id, quantity) pairs");
  if (deck_pool_seq == NULL) {
    return -1;
  }

  const Py_ssize_t deck_pool_count = PySequence_Fast_GET_SIZE(deck_pool_seq);
  if (deck_pool_count <= 0) {
    Py_DECREF(deck_pool_seq);
    PyErr_SetString(PyExc_ValueError, "deck_pool must contain at least one deck");
    return -1;
  }

  free_training_deck_pool(env);
  env->deck_pool = calloc((size_t)deck_pool_count, sizeof(TrainingDeckSpec));
  if (env->deck_pool == NULL) {
    Py_DECREF(deck_pool_seq);
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate training deck pool");
    return -1;
  }
  env->deck_pool_count = (size_t)deck_pool_count;
  reset_current_deck_indices(env);

  for (Py_ssize_t deck_index = 0; deck_index < deck_pool_count; ++deck_index) {
    PyObject *deck_obj = PySequence_Fast_GET_ITEM(deck_pool_seq, deck_index);
    PyObject *deck_seq = PySequence_Fast(
        deck_obj,
        "Each deck in deck_pool must be a sequence of (card_id, quantity) pairs");
    if (deck_seq == NULL) {
      free_training_deck_pool(env);
      Py_DECREF(deck_pool_seq);
      return -1;
    }

    const Py_ssize_t card_count = PySequence_Fast_GET_SIZE(deck_seq);
    if (card_count <= 0) {
      Py_DECREF(deck_seq);
      free_training_deck_pool(env);
      Py_DECREF(deck_pool_seq);
      PyErr_Format(PyExc_ValueError, "deck_pool[%zd] must contain at least one card entry", deck_index);
      return -1;
    }

    CardInfo *cards = calloc((size_t)card_count, sizeof(CardInfo));
    if (cards == NULL) {
      Py_DECREF(deck_seq);
      free_training_deck_pool(env);
      Py_DECREF(deck_pool_seq);
      PyErr_SetString(PyExc_MemoryError, "Failed to allocate deck card array");
      return -1;
    }

    int total_cards = 0;
    for (Py_ssize_t card_index = 0; card_index < card_count; ++card_index) {
      PyObject *card_entry_obj = PySequence_Fast_GET_ITEM(deck_seq, card_index);
      PyObject *card_entry_seq = PySequence_Fast(
          card_entry_obj,
          "Each deck entry must be a (card_id, quantity) pair");
      if (card_entry_seq == NULL) {
        free(cards);
        Py_DECREF(deck_seq);
        free_training_deck_pool(env);
        Py_DECREF(deck_pool_seq);
        return -1;
      }

      if (PySequence_Fast_GET_SIZE(card_entry_seq) != 2) {
        Py_DECREF(card_entry_seq);
        free(cards);
        Py_DECREF(deck_seq);
        free_training_deck_pool(env);
        Py_DECREF(deck_pool_seq);
        PyErr_Format(
            PyExc_ValueError,
            "deck_pool[%zd][%zd] must contain exactly 2 items: (card_id, quantity)",
            deck_index,
            card_index);
        return -1;
      }

      PyObject *card_code_obj = PySequence_Fast_GET_ITEM(card_entry_seq, 0);
      PyObject *quantity_obj = PySequence_Fast_GET_ITEM(card_entry_seq, 1);
      if (!PyUnicode_Check(card_code_obj)) {
        Py_DECREF(card_entry_seq);
        free(cards);
        Py_DECREF(deck_seq);
        free_training_deck_pool(env);
        Py_DECREF(deck_pool_seq);
        PyErr_Format(PyExc_TypeError, "deck_pool[%zd][%zd] card_id must be a string", deck_index, card_index);
        return -1;
      }

      const char *card_code = PyUnicode_AsUTF8(card_code_obj);
      if (card_code == NULL) {
        Py_DECREF(card_entry_seq);
        free(cards);
        Py_DECREF(deck_seq);
        free_training_deck_pool(env);
        Py_DECREF(deck_pool_seq);
        return -1;
      }

      CardDefId card_id;
      if (lookup_card_def_id(card_code, &card_id) != 0) {
        Py_DECREF(card_entry_seq);
        free(cards);
        Py_DECREF(deck_seq);
        free_training_deck_pool(env);
        Py_DECREF(deck_pool_seq);
        return -1;
      }

      const int quantity = parse_card_quantity(quantity_obj, card_code);
      if (quantity < 0) {
        Py_DECREF(card_entry_seq);
        free(cards);
        Py_DECREF(deck_seq);
        free_training_deck_pool(env);
        Py_DECREF(deck_pool_seq);
        return -1;
      }

      cards[card_index].card_id = card_id;
      cards[card_index].card_count = quantity;
      total_cards += quantity;
      Py_DECREF(card_entry_seq);
    }

    Py_DECREF(deck_seq);

    const int expected_total_cards = REQUIRED_DECK_SIZE + REQUIRED_LEADER_SIZE +
                                     REQUIRED_GATE_SIZE + REQUIRED_IKZ_PILE_SIZE;
    if (total_cards != expected_total_cards) {
      free(cards);
      free_training_deck_pool(env);
      Py_DECREF(deck_pool_seq);
      PyErr_Format(
          PyExc_ValueError,
          "deck_pool[%zd] resolves to %d total cards; expected %d",
          deck_index,
          total_cards,
          expected_total_cards);
      return -1;
    }

    env->deck_pool[deck_index].cards = cards;
    env->deck_pool[deck_index].card_count = (size_t)card_count;
  }

  Py_DECREF(deck_pool_seq);
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

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
  env->seed = unpack(kwargs, "seed");
  if (PyErr_Occurred()) {
    return -1;
  }

  PyObject *deck_pool_obj = PyDict_GetItemString(kwargs, "deck_pool");
  if (deck_pool_obj != NULL && load_training_deck_pool(env, deck_pool_obj) != 0) {
    return -1;
  }

  init(env);
  if (env->engine == NULL) {
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
    assign_to_dict(dict, "n", log->n);
    return 0;
}
