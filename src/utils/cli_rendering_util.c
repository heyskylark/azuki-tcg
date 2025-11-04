#include "utils/cli_rendering_util.h"

#include <ncurses.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#define MIN_BOARD_HEIGHT 8
#define MIN_LOG_HEIGHT 4
#define MIN_INFO_HEIGHT 3
#define MIN_INPUT_HEIGHT 3
#define CLI_LOG_MAX_LINES 256
#define CLI_LOG_LINE_LENGTH 256

static WINDOW *board_win = NULL;
static WINDOW *log_win = NULL;
static WINDOW *info_win = NULL;
static WINDOW *input_win = NULL;
static int screen_rows = 0;
static int screen_cols = 0;

static ObservationData last_observation = {0};
static GameState last_state = {0};
static bool has_last_state = false;
static char log_lines[CLI_LOG_MAX_LINES][CLI_LOG_LINE_LENGTH];
static size_t log_line_count = 0;
static size_t log_line_head = 0;
static void refresh_log_window(void);

static void append_log_line(const char *text) {
  if (!text) {
    return;
  }
  size_t index = log_line_head;
  strncpy(log_lines[index], text, CLI_LOG_LINE_LENGTH - 1);
  log_lines[index][CLI_LOG_LINE_LENGTH - 1] = '\0';
  size_t len = strlen(log_lines[index]);
  while (len > 0 && (log_lines[index][len - 1] == '\n' || log_lines[index][len - 1] == '\r')) {
    log_lines[index][--len] = '\0';
  }
  log_line_head = (log_line_head + 1) % CLI_LOG_MAX_LINES;
  if (log_line_count < CLI_LOG_MAX_LINES) {
    log_line_count++;
  }
}

static void render_logs(WINDOW *win) {
  if (!win) {
    return;
  }

  int max_inner_rows = getmaxy(win) - 2;
  if (max_inner_rows < 1) {
    return;
  }

  int lines_to_show = (int)((log_line_count < (size_t)max_inner_rows) ? log_line_count : (size_t)max_inner_rows);
  if (lines_to_show <= 0) {
    mvwprintw(win, 1, 2, "(no log messages)");
    return;
  }

  size_t start_index = (log_line_head + CLI_LOG_MAX_LINES - (size_t)lines_to_show) % CLI_LOG_MAX_LINES;
  for (int i = 0; i < lines_to_show; ++i) {
    size_t idx = (start_index + i) % CLI_LOG_MAX_LINES;
    mvwprintw(win, i + 1, 2, "%s", log_lines[idx]);
  }
}

static void refresh_log_window(void) {
  if (!log_win) {
    return;
  }
  werase(log_win);
  box(log_win, 0, 0);
  render_logs(log_win);
  wrefresh(log_win);
}

static void destroy_windows(void) {
  if (board_win) {
    delwin(board_win);
    board_win = NULL;
  }
  if (log_win) {
    delwin(log_win);
    log_win = NULL;
  }
  if (info_win) {
    delwin(info_win);
    info_win = NULL;
  }
  if (input_win) {
    delwin(input_win);
    input_win = NULL;
  }
}

static void ensure_windows(void) {
  int rows = 0;
  int cols = 0;
  getmaxyx(stdscr, rows, cols);

  if (rows <= 0 || cols <= 0) {
    return;
  }

  bool size_changed = rows != screen_rows || cols != screen_cols;
  if (!size_changed && board_win && log_win && info_win && input_win) {
    return;
  }

  destroy_windows();

  screen_rows = rows;
  screen_cols = cols;

  int heights[4] = { MIN_BOARD_HEIGHT, MIN_LOG_HEIGHT, MIN_INFO_HEIGHT, MIN_INPUT_HEIGHT };
  const int mins[4] = { 1, 1, 1, 2 };
  int total = heights[0] + heights[1] + heights[2] + heights[3];

  if (rows >= total) {
    heights[0] += rows - total;
  } else {
    int deficit = total - rows;
    while (deficit > 0) {
      bool reduced = false;
      for (int i = 3; i >= 0 && deficit > 0; --i) {
        if (heights[i] > mins[i]) {
          heights[i]--;
          deficit--;
          reduced = true;
        }
      }
      if (!reduced) {
        break;
      }
    }
    total = heights[0] + heights[1] + heights[2] + heights[3];
    if (total > rows) {
      heights[0] -= total - rows;
      if (heights[0] < 1) {
        heights[0] = 1;
      }
    }
  }

  int starts[4] = {0};
  for (int i = 1; i < 4; ++i) {
    starts[i] = starts[i - 1] + heights[i - 1];
  }

  board_win = newwin(heights[0], cols, starts[0], 0);
  log_win = newwin(heights[1], cols, starts[1], 0);
  info_win = newwin(heights[2], cols, starts[2], 0);
  input_win = newwin(heights[3], cols, starts[3], 0);

  if (board_win) {
    keypad(board_win, FALSE);
  }
  if (log_win) {
    keypad(log_win, FALSE);
  }
  if (info_win) {
    keypad(info_win, FALSE);
  }
  if (input_win) {
    keypad(input_win, TRUE);
  }
}

static const char *phase_to_string(Phase phase) {
  switch (phase) {
    case PHASE_PREGAME_MULLIGAN:
      return "Pregame Mulligan";
    case PHASE_START_OF_TURN:
      return "Start of Turn";
    case PHASE_MAIN:
      return "Main";
    case PHASE_COMBAT_DECLARED:
      return "Combat Declared";
    case PHASE_RESPONSE_WINDOW:
      return "Response Window";
    case PHASE_COMBAT_RESOLVE:
      return "Combat Resolve";
    case PHASE_END_TURN:
      return "End Turn";
    case PHASE_END_MATCH:
      return "End Match";
    default:
      return "Unknown";
  }
}

static const char *card_code_or_placeholder(const CardId *id) {
  if (!id || !id->code || id->code[0] == '\0') {
    return "<?>"; 
  }
  return id->code;
}

static char tap_state_char(const TapState *tap_state) {
  if (!tap_state) {
    return '-';
  }
  return tap_state->tapped ? 'T' : 'U';
}

static void format_leader_summary(const LeaderCardObservationData *leader, char *buffer, size_t length) {
  const char *code = card_code_or_placeholder(&leader->id);
  char tap = tap_state_char(&leader->tap_state);
  snprintf(buffer, length, "%s[%d/%d]%c", code, leader->cur_stats.cur_atk, leader->cur_stats.cur_hp, tap);
}

static void format_gate_summary(const GateCardObservationData *gate, char *buffer, size_t length) {
  const char *code = card_code_or_placeholder(&gate->id);
  char tap = tap_state_char(&gate->tap_state);
  snprintf(buffer, length, "%s%c", code, tap);
}

static void format_card_summary(const CardObservationData *card, char *buffer, size_t length) {
  const char *code = card_code_or_placeholder(&card->id);
  char tap = tap_state_char(&card->tap_state);
  if (card->has_cur_stats) {
    snprintf(buffer, length, "%s[%d/%d]%c", code, card->cur_stats.cur_atk, card->cur_stats.cur_hp, tap);
    return;
  }
  if (card->has_gate_points) {
    snprintf(buffer, length, "%s{GP:%u}%c", code, card->gate_points.gate_points, tap);
    return;
  }
  snprintf(buffer, length, "%s%c", code, tap);
}

static void format_ikz_card_summary(const IKZCardObservationData *card, char *buffer, size_t length) {
  const char *code = card_code_or_placeholder(&card->id);
  char tap = tap_state_char(&card->tap_state);
  snprintf(buffer, length, "%s%c", code, tap);
}

static size_t card_observation_count(const CardObservationData *cards, size_t max_count) {
  if (!cards) {
    return 0;
  }
  size_t count = 0;
  for (size_t i = 0; i < max_count; i++) {
    if (!cards[i].id.code) {
      break;
    }
    count++;
  }
  return count;
}

static size_t ikz_observation_count(const IKZCardObservationData *cards, size_t max_count) {
  if (!cards) {
    return 0;
  }
  size_t count = 0;
  for (size_t i = 0; i < max_count; i++) {
    if (!cards[i].id.code) {
      break;
    }
    count++;
  }
  return count;
}

static void draw_card_group(WINDOW *win, int row, const char *label, const CardObservationData *cards, size_t count) {
  if (!win) {
    return;
  }
  int max_cols = getmaxx(win);
  if (max_cols <= 4) {
    return;
  }
  mvwprintw(win, row, 2, "%s", label);
  int label_len = (int)strlen(label);
  int col = 2 + label_len + 1;
  int inner_limit = max_cols - 3;
  for (size_t i = 0; i < count; i++) {
    char summary[64];
    format_card_summary(&cards[i], summary, sizeof summary);
    int len = (int)strlen(summary);
    if (col + len >= inner_limit) {
      if (col + 3 < inner_limit) {
        mvwprintw(win, row, col, "...");
      }
      break;
    }
    mvwprintw(win, row, col, "%s", summary);
    col += len + 1;
  }
}

static void draw_ikz_group(WINDOW *win, int row, const char *label, const IKZCardObservationData *cards, size_t count) {
  if (!win) {
    return;
  }
  int max_cols = getmaxx(win);
  if (max_cols <= 4) {
    return;
  }
  mvwprintw(win, row, 2, "%s", label);
  int label_len = (int)strlen(label);
  int col = 2 + label_len + 1;
  int inner_limit = max_cols - 3;
  for (size_t i = 0; i < count; i++) {
    char summary[64];
    format_ikz_card_summary(&cards[i], summary, sizeof summary);
    int len = (int)strlen(summary);
    if (col + len >= inner_limit) {
      if (col + 3 < inner_limit) {
        mvwprintw(win, row, col, "...");
      }
      break;
    }
    mvwprintw(win, row, col, "%s", summary);
    col += len + 1;
  }
}

static void render_board(WINDOW *win, const ObservationData *observation, const GameState *gs) {
  if (!win || !observation || !gs) {
    return;
  }

  const int inner_height = getmaxy(win) - 2;
  if (inner_height <= 0) {
    return;
  }

  int row = 1;
  int max_rows = getmaxy(win) - 2;

  mvwprintw(win, row++, 2, "Opponent Board");
  if (row > max_rows) {
    return;
  }

  {
    char leader_summary[64];
    format_leader_summary(&observation->opponent_observation_data.leader, leader_summary, sizeof leader_summary);
    mvwprintw(win, row++, 4, "Leader: %s", leader_summary);
  }
  if (row > max_rows) {
    return;
  }

  {
    char gate_summary[64];
    format_gate_summary(&observation->opponent_observation_data.gate, gate_summary, sizeof gate_summary);
    mvwprintw(win, row++, 4, "Gate: %s", gate_summary);
  }
  if (row > max_rows) {
    return;
  }

  size_t opp_garden_count = card_observation_count(observation->opponent_observation_data.garden, GARDEN_SIZE);
  draw_card_group(win, row++, "Garden:", observation->opponent_observation_data.garden, opp_garden_count);
  if (row > max_rows) {
    return;
  }

  size_t opp_alley_count = card_observation_count(observation->opponent_observation_data.alley, ALLEY_SIZE);
  draw_card_group(win, row++, "Alley:", observation->opponent_observation_data.alley, opp_alley_count);
  if (row > max_rows) {
    return;
  }

  size_t opp_ikz_area_count = ikz_observation_count(observation->opponent_observation_data.ikz_area, IKZ_AREA_SIZE);
  draw_ikz_group(win, row++, "IKZ Area:", observation->opponent_observation_data.ikz_area, opp_ikz_area_count);
  if (row > max_rows) {
    return;
  }

  mvwprintw(win, row++, 4, "Hand: %u  IKZ Pile: %u  Discard: %u  IKZ Token: %s",
            observation->opponent_observation_data.hand_count,
            observation->opponent_observation_data.ikz_pile_count,
            observation->opponent_observation_data.discard_count,
            observation->opponent_observation_data.has_ikz_token ? "Yes" : "No");
  if (row > max_rows) {
    return;
  }

  row++;
  if (row > max_rows) {
    return;
  }

  mvwprintw(win, row++, 2, "Your Board");
  if (row > max_rows) {
    return;
  }

  {
    char leader_summary[64];
    format_leader_summary(&observation->my_observation_data.leader, leader_summary, sizeof leader_summary);
    mvwprintw(win, row++, 4, "Leader: %s", leader_summary);
  }
  if (row > max_rows) {
    return;
  }

  {
    char gate_summary[64];
    format_gate_summary(&observation->my_observation_data.gate, gate_summary, sizeof gate_summary);
    mvwprintw(win, row++, 4, "Gate: %s", gate_summary);
  }
  if (row > max_rows) {
    return;
  }

  size_t my_garden_count = card_observation_count(observation->my_observation_data.garden, GARDEN_SIZE);
  draw_card_group(win, row++, "Garden:", observation->my_observation_data.garden, my_garden_count);
  if (row > max_rows) {
    return;
  }

  size_t my_alley_count = card_observation_count(observation->my_observation_data.alley, ALLEY_SIZE);
  draw_card_group(win, row++, "Alley:", observation->my_observation_data.alley, my_alley_count);
  if (row > max_rows) {
    return;
  }

  size_t my_ikz_area_count = ikz_observation_count(observation->my_observation_data.ikz_area, IKZ_AREA_SIZE);
  draw_ikz_group(win, row++, "IKZ Area:", observation->my_observation_data.ikz_area, my_ikz_area_count);
  if (row > max_rows) {
    return;
  }

  size_t my_hand_count = card_observation_count(observation->my_observation_data.hand, MAX_HAND_SIZE);
  draw_card_group(win, row++, "Hand:", observation->my_observation_data.hand, my_hand_count);
  if (row > max_rows) {
    return;
  }

  mvwprintw(win, row++, 4, "IKZ Pile: %u  Discard: %u  IKZ Token: %s",
            observation->my_observation_data.ikz_pile_count,
            observation->my_observation_data.discard_count,
            observation->my_observation_data.has_ikz_token ? "Yes" : "No");
}

static void render_info(WINDOW *win, const GameState *gs) {
  if (!win || !gs) {
    return;
  }
  int row = 1;
  mvwprintw(win, row++, 2, "Phase: %s", phase_to_string(gs->phase));
  mvwprintw(win, row++, 2, "Active Player: %d", gs->active_player_index);
  mvwprintw(win, row++, 2, "Response Window: %u", gs->response_window);
  if (gs->winner >= 0) {
    mvwprintw(win, row++, 2, "Winner: Player %d", gs->winner);
  } else {
    mvwprintw(win, row++, 2, "Winner: (in progress)");
  }
}

void cli_render_init(void) {
  initscr();
  cbreak();
  noecho();
  keypad(stdscr, TRUE);
  curs_set(0);
  ensure_windows();
  refresh_log_window();
  if (input_win) {
    keypad(input_win, TRUE);
  }
  has_last_state = false;
  refresh();
}

void cli_render_shutdown(void) {
  destroy_windows();
  has_last_state = false;
  screen_rows = 0;
  screen_cols = 0;
  endwin();
}

void cli_render_draw(const ObservationData *observation, const GameState *gs) {
  if (!observation || !gs) {
    return;
  }

  ensure_windows();

  if (!board_win || !log_win || !info_win || !input_win) {
    return;
  }

  last_observation = *observation;
  last_state = *gs;
  has_last_state = true;

  werase(board_win);
  box(board_win, 0, 0);
  render_board(board_win, observation, gs);
  wrefresh(board_win);

  refresh_log_window();

  werase(info_win);
  box(info_win, 0, 0);
  render_info(info_win, gs);
  wrefresh(info_win);

  werase(input_win);
  box(input_win, 0, 0);
  mvwprintw(input_win, 1, 2, "No input required.");
  wrefresh(input_win);

  curs_set(0);
}

bool cli_render_prompt_user_action(int active_player_index, const char *message, char *buffer, size_t buffer_length) {
  if (!buffer || buffer_length == 0) {
    return false;
  }

  ensure_windows();

  if (!input_win) {
    return false;
  }

  while (true) {
    ensure_windows();
    if (!input_win) {
      return false;
    }
    refresh_log_window();

    werase(input_win);
    box(input_win, 0, 0);

    int max_inner_rows = getmaxy(input_win) - 2;
    int current_row = 1;
    if (message && message[0] != '\0' && current_row <= max_inner_rows) {
      mvwprintw(input_win, current_row++, 2, "%s", message);
    }
    if (current_row <= max_inner_rows) {
      mvwprintw(input_win, current_row++, 2, "Awaiting user action [Player %d] (type,p0,p1,p2):", active_player_index);
    }
    if (current_row > max_inner_rows) {
      current_row = max_inner_rows;
    }
    if (current_row < 1) {
      current_row = 1;
    }
    wmove(input_win, current_row, 2);
    wclrtoeol(input_win);
    wrefresh(input_win);

    echo();
    curs_set(1);
    int result = wgetnstr(input_win, buffer, (int)buffer_length - 1);
    noecho();
    curs_set(0);

    if (result == ERR) {
      if (is_term_resized(screen_rows, screen_cols)) {
        if (has_last_state) {
          cli_render_draw(&last_observation, &last_state);
        } else {
          refresh();
        }
        continue;
      }
      return false;
    }

    size_t len = strlen(buffer);
    while (len > 0 && (buffer[len - 1] == '\n' || buffer[len - 1] == '\r')) {
      buffer[--len] = '\0';
    }
    return true;
  }
}

void cli_render_log(const char *message) {
  if (!message) {
    return;
  }
  append_log_line(message);
  refresh_log_window();
}

void cli_render_logf(const char *fmt, ...) {
  if (!fmt) {
    return;
  }
  char buffer[CLI_LOG_LINE_LENGTH];
  va_list args;
  va_start(args, fmt);
  vsnprintf(buffer, sizeof buffer, fmt, args);
  va_end(args);
  cli_render_log(buffer);
}
