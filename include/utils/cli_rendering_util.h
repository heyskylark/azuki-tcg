#ifndef AZUKI_UTILS_CLI_RENDERING_UTIL_H
#define AZUKI_UTILS_CLI_RENDERING_UTIL_H

#include <stdbool.h>
#include <stddef.h>

#include "components.h"
#include "utils/observation_util.h"

void cli_render_init(void);
void cli_render_shutdown(void);
void cli_render_draw(const ObservationData *observation, const GameState *gs);
bool cli_render_prompt_user_action(int active_player_index, const char *message, char *buffer, size_t buffer_length);
void cli_render_log(const char *message);
void cli_render_logf(const char *fmt, ...);

#endif
