#include "tcg.h"

#include <Python.h>

static PyObject* env_reset_with_decks(PyObject* self, PyObject* args);
static PyObject* vec_draft_snapshot(PyObject* self, PyObject* args);
static PyObject* vec_drain_deck_records(PyObject* self, PyObject* args);
static PyObject* vec_reset_evaluation_games(PyObject* self, PyObject* args);
static PyObject* vec_active_players(PyObject* self, PyObject* args);
static PyObject* vec_force_evaluation_truncations(PyObject* self, PyObject* args);
static PyObject* obs_struct_sizes(PyObject* self, PyObject* args);

#define MY_METHODS \
  {"env_reset_with_decks", env_reset_with_decks, METH_VARARGS, "Reset the environment with two explicit deck specs"}, \
  {"vec_draft_snapshot", vec_draft_snapshot, METH_VARARGS, "Copy a completed native draft for one environment seat"}, \
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
static int deck_spec_context(
    const TrainingDeckSpec *spec, int16_t *out_gate, int16_t *out_leader) {
  int gate_count = 0;
  int leader_count = 0;
  *out_gate = -1;
  *out_leader = -1;
  for (size_t card_index = 0; card_index < spec->card_count; ++card_index) {
    const CardDef *def =
        azk_card_def_from_id((CardDefId)spec->cards[card_index].card_id);
    if (def == NULL) {
      return -1;
    }
    if (def->type == CARD_TYPE_GATE) {
      *out_gate = (int16_t)spec->cards[card_index].card_id;
      gate_count += spec->cards[card_index].card_count;
    } else if (def->type == CARD_TYPE_LEADER) {
      *out_leader = (int16_t)spec->cards[card_index].card_id;
      leader_count += spec->cards[card_index].card_count;
    }
  }
  return gate_count == 1 && leader_count == 1 ? 0 : -1;
}

static int load_prebuilt_curriculum(
    Env *env, PyObject *groups_obj, PyObject *probability_obj) {
  if (groups_obj == NULL || groups_obj == Py_None) {
    if (probability_obj != NULL && probability_obj != Py_None) {
      PyErr_SetString(
          PyExc_ValueError,
          "prebuilt_probability requires prebuilt_deck_groups");
      return -1;
    }
    return 0;
  }
  if (env->deck_pool == NULL || env->deck_pool_count == 0) {
    PyErr_SetString(
        PyExc_ValueError, "prebuilt_deck_groups requires a non-empty deck_pool");
    return -1;
  }
  if (probability_obj == NULL || probability_obj == Py_None ||
      !PyObject_TypeCheck(probability_obj, &PyArray_Type)) {
    PyErr_SetString(
        PyExc_TypeError,
        "prebuilt_probability must be a contiguous float32 NumPy vector of length 1");
    return -1;
  }
  PyArrayObject *probability = (PyArrayObject *)probability_obj;
  if (!PyArray_ISCONTIGUOUS(probability) ||
      PyArray_TYPE(probability) != NPY_FLOAT32 ||
      PyArray_NDIM(probability) != 1 || PyArray_SIZE(probability) != 1) {
    PyErr_SetString(
        PyExc_ValueError,
        "prebuilt_probability must be a contiguous float32 NumPy vector of length 1");
    return -1;
  }
  const float initial_probability = *(float *)PyArray_DATA(probability);
  if (!(initial_probability >= 0.0f && initial_probability <= 1.0f)) {
    PyErr_SetString(PyExc_ValueError, "prebuilt_probability must be in [0, 1]");
    return -1;
  }

  /* Own immutable snapshots: iterators must not be consumed twice, and user
   * sequences must not change the allocation size between the two passes. */
  PyObject *groups = PySequence_List(groups_obj);
  if (groups == NULL) {
    return -1;
  }
  const Py_ssize_t group_count = PySequence_Fast_GET_SIZE(groups);
  if (group_count <= 0) {
    Py_DECREF(groups);
    PyErr_SetString(
        PyExc_ValueError, "prebuilt_deck_groups must contain at least one group");
    return -1;
  }
  if ((uint64_t)group_count > UINT32_MAX ||
      env->deck_pool_count > (size_t)INT_MAX) {
    Py_DECREF(groups);
    PyErr_SetString(
        PyExc_OverflowError, "prebuilt curriculum index space is too large");
    return -1;
  }

  size_t total_indices = 0;
  for (Py_ssize_t group_index = 0; group_index < group_count; ++group_index) {
    PyObject *group = PySequence_Tuple(
        PyList_GET_ITEM(groups, group_index));
    if (group == NULL) {
      Py_DECREF(groups);
      return -1;
    }
    const Py_ssize_t group_size = PySequence_Fast_GET_SIZE(group);
    if (group_size <= 0 ||
        (size_t)group_size > env->deck_pool_count - total_indices) {
      Py_DECREF(group);
      Py_DECREF(groups);
      PyErr_Format(
          PyExc_ValueError, "prebuilt_deck_groups[%zd] is empty or exceeds deck_pool size",
          group_index);
      return -1;
    }
    PyList_SetItem(groups, group_index, group);
    total_indices += (size_t)group_size;
  }

  int *flat = calloc(total_indices, sizeof(*flat));
  size_t *offsets =
      calloc((size_t)group_count + 1, sizeof(*offsets));
  bool *seen = calloc(env->deck_pool_count, sizeof(*seen));
  int16_t *context_gates = calloc((size_t)group_count, sizeof(*context_gates));
  int16_t *context_leaders =
      calloc((size_t)group_count, sizeof(*context_leaders));
  if (flat == NULL || offsets == NULL || seen == NULL ||
      context_gates == NULL || context_leaders == NULL) {
    free(flat);
    free(offsets);
    free(seen);
    free(context_gates);
    free(context_leaders);
    Py_DECREF(groups);
    PyErr_SetString(
        PyExc_MemoryError, "Failed to allocate prebuilt curriculum groups");
    return -1;
  }

  size_t cursor = 0;
  int status = 0;
  for (Py_ssize_t group_index = 0;
       group_index < group_count && status == 0; ++group_index) {
    PyObject *group = PyList_GET_ITEM(groups, group_index);
    offsets[group_index] = cursor;
    const Py_ssize_t group_size = PySequence_Fast_GET_SIZE(group);
    int16_t expected_gate = -1;
    int16_t expected_leader = -1;
    for (Py_ssize_t item = 0; item < group_size; ++item) {
      PyObject *index_obj = PySequence_Fast_GET_ITEM(group, item);
      if (!PyLong_Check(index_obj) || PyBool_Check(index_obj)) {
        PyErr_Format(
            PyExc_TypeError,
            "prebuilt_deck_groups[%zd][%zd] must be an integer",
            group_index, item);
        status = -1;
        break;
      }
      const long index = PyLong_AsLong(index_obj);
      if (PyErr_Occurred()) {
        status = -1;
        break;
      }
      if (index < 0 || (size_t)index >= env->deck_pool_count) {
        PyErr_Format(
            PyExc_IndexError,
            "prebuilt_deck_groups[%zd][%zd]=%ld is outside deck_pool",
            group_index, item, index);
        status = -1;
        break;
      }
      if (seen[index]) {
        PyErr_Format(
            PyExc_ValueError,
            "deck_pool index %ld appears in more than one prebuilt group",
            index);
        status = -1;
        break;
      }
      int16_t gate = -1;
      int16_t leader = -1;
      if (deck_spec_context(&env->deck_pool[index], &gate, &leader) != 0) {
        PyErr_Format(
            PyExc_ValueError,
            "deck_pool[%ld] must contain exactly one gate and one leader",
            index);
        status = -1;
        break;
      }
      if (item == 0) {
        expected_gate = gate;
        expected_leader = leader;
      } else if (gate != expected_gate || leader != expected_leader) {
        PyErr_Format(
            PyExc_ValueError,
            "prebuilt_deck_groups[%zd] mixes gate/leader contexts",
            group_index);
        status = -1;
        break;
      }
      seen[index] = true;
      flat[cursor++] = (int)index;
    }
    if (status != 0) {
      break;
    }
    for (Py_ssize_t prior = 0; prior < group_index; ++prior) {
      if (context_gates[prior] == expected_gate &&
          context_leaders[prior] == expected_leader) {
        PyErr_Format(
            PyExc_ValueError,
            "prebuilt groups %zd and %zd describe the same gate/leader context",
            prior, group_index);
        status = -1;
        break;
      }
    }
    context_gates[group_index] = expected_gate;
    context_leaders[group_index] = expected_leader;
  }
  offsets[group_count] = cursor;
  Py_DECREF(groups);
  free(seen);
  free(context_gates);
  free(context_leaders);
  if (status != 0) {
    free(flat);
    free(offsets);
    return -1;
  }

  env->prebuilt_deck_indices = flat;
  env->prebuilt_group_offsets = offsets;
  env->prebuilt_group_count = (size_t)group_count;
  env->prebuilt_probability = (float *)PyArray_DATA(probability);
  return 0;
}

// Element-specialist knob: learner_element (CardElement, 0 = disabled) plus a
// caller-owned uint8 per-seat mask indexed [2 * env_index + seat]. Default
// (absent/0) leaves every field zero so RNG streams are untouched.
static int load_learner_element(Env *env, PyObject *kwargs) {
  PyObject *element_obj = PyDict_GetItemString(kwargs, "learner_element");
  if (element_obj == NULL || element_obj == Py_None) {
    return 0;
  }
  const long element = PyLong_AsLong(element_obj);
  if (PyErr_Occurred()) {
    return -1;
  }
  if (element == CARD_ELEMENT_NORMAL) {
    return 0;
  }
  if (element != CARD_ELEMENT_FIRE && element != CARD_ELEMENT_WATER &&
      element != CARD_ELEMENT_EARTH && element != CARD_ELEMENT_LIGHTNING) {
    PyErr_Format(PyExc_ValueError, "learner_element %ld is not a playable element", element);
    return -1;
  }
  if (!env->deck_building) {
    PyErr_SetString(PyExc_ValueError, "learner_element requires deck_building=True");
    return -1;
  }
  PyObject *mask_obj = PyDict_GetItemString(kwargs, "learner_seat_mask");
  PyObject *index_obj = PyDict_GetItemString(kwargs, "env_index");
  if (mask_obj == NULL || !PyObject_TypeCheck(mask_obj, &PyArray_Type) ||
      index_obj == NULL) {
    PyErr_SetString(
        PyExc_ValueError,
        "learner_element requires a learner_seat_mask uint8 array and env_index");
    return -1;
  }
  PyArrayObject *mask = (PyArrayObject *)mask_obj;
  const long env_index = PyLong_AsLong(index_obj);
  if (PyErr_Occurred()) {
    return -1;
  }
  if (!PyArray_ISCONTIGUOUS(mask) || PyArray_TYPE(mask) != NPY_UINT8 ||
      env_index < 0 ||
      PyArray_SIZE(mask) < (npy_intp)(MAX_PLAYERS_PER_MATCH * (env_index + 1))) {
    PyErr_SetString(
        PyExc_ValueError,
        "learner_seat_mask must be a contiguous uint8 array covering every env seat");
    return -1;
  }
  int gate_matches = 0;
  for (int i = 0; i < g_draft_catalog.gate_count; ++i) {
    if (draft_gate_has_element(g_draft_catalog.gate_def_ids[i], (int8_t)element)) {
      gate_matches++;
    }
  }
  if (gate_matches == 0) {
    PyErr_Format(PyExc_ValueError, "draft catalog has no gate of learner_element %ld", element);
    return -1;
  }
  PyObject *decks_obj = PyDict_GetItemString(kwargs, "learner_prebuilt_deck_indices");
  if (env->prebuilt_group_count > 0) {
    PyObject *decks = decks_obj == NULL
        ? NULL
        : PySequence_Fast(decks_obj, "learner_prebuilt_deck_indices must be a sequence");
    if (decks == NULL || PySequence_Fast_GET_SIZE(decks) == 0) {
      Py_XDECREF(decks);
      if (!PyErr_Occurred()) {
        PyErr_SetString(
            PyExc_ValueError,
            "learner_element with prebuilt decks requires non-empty learner_prebuilt_deck_indices");
      }
      return -1;
    }
    const Py_ssize_t count = PySequence_Fast_GET_SIZE(decks);
    int *learner_decks = (int *)calloc((size_t)count, sizeof(int));
    if (learner_decks == NULL) {
      Py_DECREF(decks);
      PyErr_SetString(PyExc_MemoryError, "Failed to allocate learner prebuilt decks");
      return -1;
    }
    for (Py_ssize_t item = 0; item < count; ++item) {
      const long index = PyLong_AsLong(PySequence_Fast_GET_ITEM(decks, item));
      bool in_curriculum = false;
      for (size_t flat = 0; !PyErr_Occurred() &&
                            flat < env->prebuilt_group_offsets[env->prebuilt_group_count];
           ++flat) {
        in_curriculum = in_curriculum || env->prebuilt_deck_indices[flat] == (int)index;
      }
      const int16_t gate = in_curriculum ? draft_spec_gate(&env->deck_pool[index]) : -1;
      if (PyErr_Occurred() || gate < 0 || !draft_gate_has_element(gate, (int8_t)element)) {
        free(learner_decks);
        Py_DECREF(decks);
        if (!PyErr_Occurred()) {
          PyErr_Format(
              PyExc_ValueError,
              "learner_prebuilt_deck_indices[%zd]=%ld is not a prebuilt deck of learner_element %ld",
              item, index, element);
        }
        return -1;
      }
      learner_decks[item] = (int)index;
    }
    Py_DECREF(decks);
    env->prebuilt_learner_decks = learner_decks;
    env->prebuilt_learner_deck_count = (size_t)count;
  } else if (decks_obj != NULL && decks_obj != Py_None) {
    PyErr_SetString(
        PyExc_ValueError, "learner_prebuilt_deck_indices requires prebuilt_deck_groups");
    return -1;
  }
  env->learner_element = (int8_t)element;
  env->learner_seat_mask =
      (const uint8_t *)PyArray_DATA(mask) + MAX_PLAYERS_PER_MATCH * env_index;
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
    env->evaluation_other_deck_index = -1;
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
  PyObject *prebuilt_groups_obj =
      PyDict_GetItemString(kwargs, "prebuilt_deck_groups");
  PyObject *prebuilt_probability_obj =
      PyDict_GetItemString(kwargs, "prebuilt_probability");
  if ((prebuilt_groups_obj != NULL || prebuilt_probability_obj != NULL) &&
      !env->deck_building) {
    PyErr_SetString(
        PyExc_ValueError,
        "prebuilt curriculum requires deck_building=True");
    free_training_deck_pool(env);
    return -1;
  }
  if (load_prebuilt_curriculum(
          env, prebuilt_groups_obj, prebuilt_probability_obj) != 0) {
    free_training_deck_pool(env);
    return -1;
  }
  if (load_learner_element(env, kwargs) != 0) {
    free_training_deck_pool(env);
    return -1;
  }
  PyObject* reward_telemetry_obj =
      PyDict_GetItemString(kwargs, "reward_telemetry");
  if (reward_telemetry_obj != NULL && PyObject_IsTrue(reward_telemetry_obj)) {
    if (!env->deck_building) {
      PyErr_SetString(
          PyExc_ValueError,
          "reward_telemetry requires deck_building=True");
      free_training_deck_pool(env);
      return -1;
    }
    env->reward_telemetry.enabled = true;
  }
  PyObject* reward_scales_obj =
      PyDict_GetItemString(kwargs, "reward_scales");
  if (reward_scales_obj != NULL) {
    if (!PyObject_TypeCheck(reward_scales_obj, &PyArray_Type)) {
      PyErr_SetString(PyExc_TypeError, "reward_scales must be a NumPy array");
      free_training_deck_pool(env);
      return -1;
    }
    PyArrayObject* reward_scales = (PyArrayObject*)reward_scales_obj;
    if (!PyArray_ISCONTIGUOUS(reward_scales) ||
        PyArray_TYPE(reward_scales) != NPY_FLOAT32 ||
        PyArray_SIZE(reward_scales) != 2) {
      PyErr_SetString(
          PyExc_ValueError,
          "reward_scales must be a contiguous float32 vector of length 2");
      free_training_deck_pool(env);
      return -1;
    }
    env->reward_scales = (float*)PyArray_DATA(reward_scales);
  }
  env->pbrs_gamma = 0.99f;
  PyObject* pbrs_mode_obj = PyDict_GetItemString(kwargs, "pbrs_mode");
  if (pbrs_mode_obj != NULL) {
    const char* pbrs_mode = PyUnicode_AsUTF8(pbrs_mode_obj);
    if (pbrs_mode == NULL) {
      free_training_deck_pool(env);
      return -1;
    }
    if (strcmp(pbrs_mode, "legacy") == 0) {
      env->proper_pbrs = false;
      env->pbrs_terminal_closure = false;
    } else if (strcmp(pbrs_mode, "discounted") == 0) {
      env->proper_pbrs = true;
      env->pbrs_terminal_closure = true;
    } else {
      PyErr_SetString(
          PyExc_ValueError,
          "pbrs_mode must be 'legacy' or 'discounted'");
      free_training_deck_pool(env);
      return -1;
    }
  }
  PyObject* pbrs_gamma_obj = PyDict_GetItemString(kwargs, "pbrs_gamma");
  if (pbrs_gamma_obj != NULL) {
    const double gamma = PyFloat_AsDouble(pbrs_gamma_obj);
    if (PyErr_Occurred()) {
      free_training_deck_pool(env);
      return -1;
    }
    if (!(gamma >= 0.0 && gamma <= 1.0)) {
      PyErr_SetString(PyExc_ValueError, "pbrs_gamma must be in [0, 1]");
      free_training_deck_pool(env);
      return -1;
    }
    env->pbrs_gamma = (float)gamma;
  }
  PyObject* pbrs_closure_obj =
      PyDict_GetItemString(kwargs, "pbrs_terminal_closure");
  if (pbrs_closure_obj != NULL && pbrs_closure_obj != Py_None) {
    const int closure = PyObject_IsTrue(pbrs_closure_obj);
    if (closure < 0) {
      free_training_deck_pool(env);
      return -1;
    }
    if (env->proper_pbrs && !closure) {
      PyErr_SetString(
          PyExc_ValueError,
          "pbrs_mode='discounted' requires terminal closure");
      free_training_deck_pool(env);
      return -1;
    }
    if (!env->proper_pbrs && closure) {
      PyErr_SetString(
          PyExc_ValueError,
          "pbrs_terminal_closure requires pbrs_mode='discounted'");
      free_training_deck_pool(env);
      return -1;
    }
    env->pbrs_terminal_closure = closure != 0;
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
    assign_to_dict(dict, "potential_reward_scale", log->potential_reward_scale);
    assign_to_dict(dict, "exploration_reward_scale", log->exploration_reward_scale);
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
    assign_to_dict(dict, "p0_gate_ability_outcomes", log->p0_gate_ability_outcomes);
    assign_to_dict(dict, "p1_gate_ability_outcomes", log->p1_gate_ability_outcomes);
    assign_to_dict(dict, "p0_leader_ability_outcomes", log->p0_leader_ability_outcomes);
    assign_to_dict(dict, "p1_leader_ability_outcomes", log->p1_leader_ability_outcomes);
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

static PyObject* vec_draft_snapshot(PyObject* self, PyObject* args) {
  (void)self;
  if (PyTuple_Size(args) != 3) {
    PyErr_SetString(
        PyExc_TypeError,
        "vec_draft_snapshot requires handle, env_index, and player_index");
    return NULL;
  }
  VecEnv* vec = unpack_vecenv(args);
  if (vec == NULL) {
    return NULL;
  }
  const long env_index = PyLong_AsLong(PyTuple_GetItem(args, 1));
  const long player_index = PyLong_AsLong(PyTuple_GetItem(args, 2));
  if (PyErr_Occurred()) {
    return NULL;
  }
  if (env_index < 0 || env_index >= vec->num_envs) {
    PyErr_Format(PyExc_IndexError, "Draft snapshot env index %ld is out of range",
                 env_index);
    return NULL;
  }
  if (player_index < 0 || player_index >= MAX_PLAYERS_PER_MATCH) {
    PyErr_Format(PyExc_IndexError,
                 "Draft snapshot player index %ld is out of range",
                 player_index);
    return NULL;
  }

  AzkDraftSnapshot snapshot;
  if (!c_draft_snapshot(vec->envs[env_index], (int)player_index, &snapshot)) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "Draft snapshot is unavailable until a native deck-building draft has completed");
    return NULL;
  }
  PyObject* main_cards = PyList_New(REQUIRED_DECK_SIZE);
  if (main_cards == NULL) {
    return NULL;
  }
  for (int index = 0; index < REQUIRED_DECK_SIZE; ++index) {
    PyObject* card_id = PyLong_FromLong(snapshot.main_card_def_ids[index]);
    if (card_id == NULL) {
      Py_DECREF(main_cards);
      return NULL;
    }
    PyList_SET_ITEM(main_cards, index, card_id);
  }
  return Py_BuildValue(
      "{s:i,s:i,s:i,s:N}",
      "gate", (int)snapshot.gate_card_def_id,
      "leader", (int)snapshot.leader_card_def_id,
      "main_count", (int)snapshot.main_count,
      "main", main_cards);
}

static bool reward_stats_active(const AzkRewardComponentStats* stats) {
  return stats->raw_positive_count != 0 || stats->raw_negative_count != 0 ||
         stats->scaled_positive_count != 0 ||
         stats->scaled_negative_count != 0 ||
         stats->raw_sum != 0.0f || stats->scaled_sum != 0.0f;
}

static PyObject* build_reward_stat_entries(
    const AzkRewardComponentStats* components) {
  PyObject* entries = PyList_New(0);
  if (entries == NULL) {
    return NULL;
  }
  for (int component = 0; component < AZK_REWARD_COMPONENT_COUNT;
       ++component) {
    const AzkRewardComponentStats* stats = &components[component];
    if (!reward_stats_active(stats)) {
      continue;
    }
    PyObject* entry = Py_BuildValue(
        "(iffffkkffffkk)",
        component,
        stats->raw_sum,
        stats->raw_abs_sum,
        stats->raw_discounted_sum,
        stats->raw_max_abs,
        (unsigned long)stats->raw_positive_count,
        (unsigned long)stats->raw_negative_count,
        stats->scaled_sum,
        stats->scaled_abs_sum,
        stats->scaled_discounted_sum,
        stats->scaled_max_abs,
        (unsigned long)stats->scaled_positive_count,
        (unsigned long)stats->scaled_negative_count);
    if (entry == NULL || PyList_Append(entries, entry) < 0) {
      Py_XDECREF(entry);
      Py_DECREF(entries);
      return NULL;
    }
    Py_DECREF(entry);
  }
  return entries;
}

static PyObject* build_reward_stat_slices(
    const AzkRewardComponentStats* slices, const uint32_t* step_counts,
    int slice_count) {
  PyObject* output = PyList_New(0);
  if (output == NULL) {
    return NULL;
  }
  for (int slice = 0; slice < slice_count; ++slice) {
    if (step_counts[slice] == 0) {
      continue;
    }
    PyObject* entries = build_reward_stat_entries(
        &slices[slice * AZK_REWARD_COMPONENT_COUNT]);
    if (entries == NULL) {
      Py_DECREF(output);
      return NULL;
    }
    PyObject* item = Py_BuildValue(
        "(ikN)", slice, (unsigned long)step_counts[slice], entries);
    if (item == NULL || PyList_Append(output, item) < 0) {
      Py_XDECREF(item);
      Py_DECREF(output);
      return NULL;
    }
    Py_DECREF(item);
  }
  return output;
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
          "{s:i,s:i,s:i,s:O,s:i,s:N,s:f,s:f,s:f,s:f,s:f,s:f,s:f,s:f,s:f}",
          "gate", (int)env->deck_record_gate[p],
          "original_gate", (int)env->deck_record_original_gate[p],
          "battle_gate", (int)env->deck_record_gate[p],
          "gate_swapped", env->deck_record_gate_swapped[p] ? Py_True : Py_False,
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
      assign_to_dict(player, "gate_ability_outcomes",
                     env->deck_record_behavior[p][18]);
      assign_to_dict(player, "leader_ability_outcomes",
                     env->deck_record_behavior[p][19]);
      if (env->reward_telemetry.enabled) {
        PyObject* overall = build_reward_stat_entries(
            env->reward_telemetry.overall[p]);
        PyObject* by_action = build_reward_stat_slices(
            &env->reward_telemetry.by_action[p][0][0],
            env->reward_telemetry.action_step_count[p],
            AZK_ACTION_TYPE_COUNT);
        PyObject* by_turn_bucket = build_reward_stat_slices(
            &env->reward_telemetry.by_turn_bucket[p][0][0],
            env->reward_telemetry.turn_bucket_step_count[p],
            AZK_REWARD_TURN_BUCKET_COUNT);
        if (overall == NULL || by_action == NULL || by_turn_bucket == NULL) {
          Py_XDECREF(overall);
          Py_XDECREF(by_action);
          Py_XDECREF(by_turn_bucket);
          Py_DECREF(player);
          Py_DECREF(players);
          Py_DECREF(records);
          return NULL;
        }
        PyObject* telemetry = Py_BuildValue(
            "{s:f,s:f,s:f,s:N,s:N,s:N}",
            "raw_shaping_return",
            env->reward_telemetry.raw_shaping_return[p],
            "scaled_shaping_return",
            env->reward_telemetry.scaled_shaping_return[p],
            "terminal_return",
            env->reward_telemetry.terminal_return[p],
            "overall", overall,
            "by_action", by_action,
            "by_turn_bucket", by_turn_bucket);
        if (telemetry == NULL) {
          Py_DECREF(player);
          Py_DECREF(players);
          Py_DECREF(records);
          return NULL;
        }
        assign_to_dict(
            telemetry, "initial_potential", env->episode_initial_phi[p]);
        assign_to_dict(
            telemetry, "initial_scaled_potential",
            env->episode_initial_scaled_phi[p]);
        assign_to_dict(
            telemetry, "final_potential", env->last_phi[p]);
        assign_to_dict(
            telemetry, "final_scaled_potential", env->last_scaled_phi[p]);
        assign_to_dict(
            telemetry, "final_discount", env->reward_telemetry.discount);
        if (PyDict_SetItemString(player, "reward_telemetry", telemetry) < 0) {
          Py_DECREF(telemetry);
          Py_DECREF(player);
          Py_DECREF(players);
          Py_DECREF(records);
          return NULL;
        }
        Py_DECREF(telemetry);
      }
      PyList_SET_ITEM(players, p, player);
    }
    PyObject* prebuilt_deck_indices = Py_BuildValue(
        "[i,i]",
        env->deck_record_prebuilt_deck_indices[0],
        env->deck_record_prebuilt_deck_indices[1]);
    if (prebuilt_deck_indices == NULL) {
      Py_DECREF(players);
      Py_DECREF(records);
      return NULL;
    }
    PyObject* record = Py_BuildValue(
        "{s:i,s:k,s:f,s:i,s:i,s:i,s:i,s:O,s:N,s:N}",
        "env_index", i,
        "seed", (unsigned long)env->deck_record_seed,
        "episode_length", env->deck_record_episode_length,
        "ref_seat", (int)env->deck_record_ref_seat,
        "ref_deck_index", (int)env->deck_record_ref_deck_index,
        "end_reason", (int)env->deck_record_end_reason,
        "starting_player", (int)env->deck_record_starting_player,
        "prebuilt", env->deck_record_prebuilt ? Py_True : Py_False,
        "prebuilt_deck_indices", prebuilt_deck_indices,
        "players", players);
    if (record == NULL) {
      Py_DECREF(records);
      return NULL;
    }
    if (env->reward_telemetry.enabled) {
      const float scale_min =
          env->reward_telemetry.shaping_step_count > 0
              ? env->reward_telemetry.shaping_scale_min
              : 0.0f;
      PyObject* telemetry = Py_BuildValue(
          "{s:f,s:f,s:f,s:f,s:f,s:k,s:f}",
          "raw_reconstruction_max_abs_error",
          env->reward_telemetry.raw_reconstruction_max_abs_error,
          "scaled_reconstruction_max_abs_error",
          env->reward_telemetry.scaled_reconstruction_max_abs_error,
          "shaping_scale_sum",
          env->reward_telemetry.shaping_scale_sum,
          "shaping_scale_min", scale_min,
          "shaping_scale_max",
          env->reward_telemetry.shaping_scale_max,
          "shaping_step_count",
          (unsigned long)env->reward_telemetry.shaping_step_count,
          "gamma",
          env->proper_pbrs ? env->pbrs_gamma
                           : AZK_REWARD_TELEMETRY_GAMMA);
      if (telemetry == NULL) {
        Py_DECREF(record);
        Py_DECREF(records);
        return NULL;
      }
      assign_to_dict(telemetry, "pbrs_gamma", env->pbrs_gamma);
      PyObject* pbrs_mode = PyUnicode_FromString(
          env->proper_pbrs ? "discounted" : "legacy");
      if (pbrs_mode == NULL ||
          PyDict_SetItemString(telemetry, "pbrs_mode", pbrs_mode) < 0 ||
          PyDict_SetItemString(
              telemetry, "pbrs_terminal_closure",
              env->pbrs_terminal_closure ? Py_True : Py_False) < 0 ||
          PyDict_SetItemString(record, "reward_telemetry", telemetry) < 0) {
        Py_XDECREF(pbrs_mode);
        Py_DECREF(telemetry);
        Py_DECREF(record);
        Py_DECREF(records);
        return NULL;
      }
      Py_DECREF(pbrs_mode);
      Py_DECREF(telemetry);
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
  // Optional 10th sequence: a fixed deck for the seat opposite ref_seat, so
  // both seats battle supplied decks (constructed-vs-constructed evaluation).
  const Py_ssize_t arg_count = PyTuple_Size(args);
  if (arg_count != 9 && arg_count != 10) {
    PyErr_SetString(
        PyExc_TypeError,
        "vec_reset_evaluation_games requires handle, indices, seeds, gate0, gate1, leader0, leader1, ref_seats, ref_decks[, other_decks]");
    return NULL;
  }
  VecEnv* vec = unpack_vecenv(args);
  if (vec == NULL) {
    return NULL;
  }

  const int sequence_count = (int)arg_count - 1;
  const char* labels[9] = {
      "indices", "seeds", "gate0", "gate1", "leader0", "leader1",
      "ref_seats", "ref_decks", "other_decks"};
  PyObject* sequences[9] = {NULL};
  for (int sequence_index = 0; sequence_index < sequence_count; ++sequence_index) {
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
  for (int sequence_index = 1; sequence_index < sequence_count; ++sequence_index) {
    if (PySequence_Fast_GET_SIZE(sequences[sequence_index]) != count) {
      for (int i = 0; i < sequence_count; ++i) {
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
    long other_deck = -1;
    if (sequence_long_at(sequences[0], item, &index) != 0 ||
        sequence_long_at(sequences[1], item, &seed) != 0 ||
        sequence_long_at(sequences[2], item, &gate0) != 0 ||
        sequence_long_at(sequences[3], item, &gate1) != 0 ||
        sequence_long_at(sequences[4], item, &leader0) != 0 ||
        sequence_long_at(sequences[5], item, &leader1) != 0 ||
        sequence_long_at(sequences[6], item, &ref_seat) != 0 ||
        sequence_long_at(sequences[7], item, &ref_deck) != 0 ||
        (sequence_count == 9 && sequence_long_at(sequences[8], item, &other_deck) != 0)) {
      for (int i = 0; i < sequence_count; ++i) {
        Py_DECREF(sequences[i]);
      }
      return NULL;
    }
    if (index < 0 || index >= vec->num_envs) {
      for (int i = 0; i < sequence_count; ++i) {
        Py_DECREF(sequences[i]);
      }
      PyErr_Format(PyExc_IndexError, "Evaluation env index %ld is out of range", index);
      return NULL;
    }
    Env* env = vec->envs[index];
    if (!env->deck_building) {
      for (int i = 0; i < sequence_count; ++i) {
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
      for (int i = 0; i < sequence_count; ++i) {
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
        for (int i = 0; i < sequence_count; ++i) {
          Py_DECREF(sequences[i]);
        }
        PyErr_Format(PyExc_ValueError,
                     "Invalid forced evaluation leaders [%ld,%ld] for gates [%ld,%ld]",
                     leader0, leader1, gate0, gate1);
        return NULL;
      }
    }
    const bool no_reference = ref_seat < 0 && ref_deck < 0;
    if ((!no_reference &&
         (ref_seat < 0 || ref_seat >= MAX_PLAYERS_PER_MATCH ||
          ref_deck < 0 || (size_t)ref_deck >= env->deck_pool_count)) ||
        (other_deck >= 0 && (no_reference || (size_t)other_deck >= env->deck_pool_count))) {
      for (int i = 0; i < sequence_count; ++i) {
        Py_DECREF(sequences[i]);
      }
      PyErr_Format(
          PyExc_ValueError,
          "Invalid evaluation reference seat/deck/other deck [%ld,%ld,%ld]",
          ref_seat, ref_deck, other_deck);
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
    env->evaluation_other_deck_index = (int16_t)(other_deck >= 0 ? other_deck : -1);
    env->deck_record_valid = false;
    env->seed = (uint32_t)seed;
    env->starter_rng_state = starter_seed_from_env_seed(env->seed);
    env->deck_rng_state = deck_seed_from_env_seed(env->seed);
    env->draft_rng_state = env->seed ^ 0x9E3779B9u;
    c_reset(env);
  }

  for (int i = 0; i < sequence_count; ++i) {
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
    refresh_observations(env);
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
