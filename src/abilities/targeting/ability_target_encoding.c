#include "abilities/targeting/ability_target_encoding.h"

#include "constants/game.h"

int azk_encode_enemy_leader_or_garden_target_index(bool is_leader,
                                                   int zone_index) {
  return is_leader ? GARDEN_SIZE : zone_index;
}

bool azk_decode_enemy_leader_or_garden_target_index(int action_index,
                                                    bool *out_is_leader,
                                                    int *out_zone_index) {
  if (action_index == GARDEN_SIZE) {
    if (out_is_leader) {
      *out_is_leader = true;
    }
    if (out_zone_index) {
      *out_zone_index = -1;
    }
    return true;
  }

  if (action_index < 0 || action_index >= GARDEN_SIZE) {
    return false;
  }

  if (out_is_leader) {
    *out_is_leader = false;
  }
  if (out_zone_index) {
    *out_zone_index = action_index;
  }
  return true;
}

int azk_encode_any_garden_target_index(bool is_enemy, int zone_index) {
  return is_enemy ? zone_index + GARDEN_SIZE : zone_index;
}

bool azk_decode_any_garden_target_index(int action_index, bool *out_is_enemy,
                                        int *out_zone_index) {
  if (action_index < 0 || action_index >= GARDEN_SIZE * 2) {
    return false;
  }

  const bool is_enemy = action_index >= GARDEN_SIZE;
  if (out_is_enemy) {
    *out_is_enemy = is_enemy;
  }
  if (out_zone_index) {
    *out_zone_index = is_enemy ? action_index - GARDEN_SIZE : action_index;
  }
  return true;
}

int azk_encode_any_leader_target_index(bool is_enemy) { return is_enemy ? 1 : 0; }

bool azk_decode_any_leader_target_index(int action_index, bool *out_is_enemy) {
  if (action_index < 0 || action_index > 1) {
    return false;
  }

  if (out_is_enemy) {
    *out_is_enemy = action_index == 1;
  }
  return true;
}

int azk_encode_friendly_garden_or_alley_target_index(bool is_alley,
                                                     int zone_index) {
  return is_alley ? zone_index + GARDEN_SIZE : zone_index;
}

bool azk_decode_friendly_garden_or_alley_target_index(int action_index,
                                                      bool *out_is_alley,
                                                      int *out_zone_index) {
  if (action_index < 0 || action_index >= GARDEN_SIZE + ALLEY_SIZE) {
    return false;
  }

  const bool is_alley = action_index >= GARDEN_SIZE;
  if (out_is_alley) {
    *out_is_alley = is_alley;
  }
  if (out_zone_index) {
    *out_zone_index = is_alley ? action_index - GARDEN_SIZE : action_index;
  }
  return true;
}

int azk_encode_any_leader_or_garden_target_index(bool is_leader, bool is_enemy,
                                                 int zone_index) {
  if (is_leader) {
    return is_enemy ? GARDEN_SIZE * 2 + 1 : GARDEN_SIZE * 2;
  }

  return azk_encode_any_garden_target_index(is_enemy, zone_index);
}

bool azk_decode_any_leader_or_garden_target_index(int action_index,
                                                  bool *out_is_leader,
                                                  bool *out_is_enemy,
                                                  int *out_zone_index) {
  if (action_index == GARDEN_SIZE * 2 || action_index == GARDEN_SIZE * 2 + 1) {
    if (out_is_leader) {
      *out_is_leader = true;
    }
    if (out_is_enemy) {
      *out_is_enemy = action_index == GARDEN_SIZE * 2 + 1;
    }
    if (out_zone_index) {
      *out_zone_index = -1;
    }
    return true;
  }

  bool is_enemy = false;
  int zone_index = -1;
  if (!azk_decode_any_garden_target_index(action_index, &is_enemy,
                                          &zone_index)) {
    return false;
  }

  if (out_is_leader) {
    *out_is_leader = false;
  }
  if (out_is_enemy) {
    *out_is_enemy = is_enemy;
  }
  if (out_zone_index) {
    *out_zone_index = zone_index;
  }
  return true;
}
