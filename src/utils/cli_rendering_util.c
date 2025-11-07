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

#define CARD_BOX_WIDTH 18
#define CARD_BOX_HEIGHT 6
#define CARD_BOX_TEXT_WIDTH (CARD_BOX_WIDTH - 2)
#define CARD_BOX_TEXT_CAPACITY (CARD_BOX_TEXT_WIDTH + 1)
#define BOARD_PAD_BUFFER_ROWS 512
#define BOARD_PAD_MAX_HEIGHT 4096

static WINDOW *board_win = NULL;
static WINDOW *log_win = NULL;
static WINDOW *info_win = NULL;
static WINDOW *input_win = NULL;
static WINDOW *board_pad = NULL;
static int board_pad_height = 0;
static int board_pad_width = 0;
static int board_scroll_offset = 0;
static int board_content_height = 0;
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

static void destroy_board_pad(void) {
  if (board_pad) {
    delwin(board_pad);
    board_pad = NULL;
  }
  board_pad_height = 0;
  board_pad_width = 0;
  board_scroll_offset = 0;
  board_content_height = 0;
}

static void ensure_board_pad(void) {
  if (!board_win) {
    destroy_board_pad();
    return;
  }

  int board_height = getmaxy(board_win);
  int board_width = getmaxx(board_win);
  if (board_height <= 2 || board_width <= 2) {
    destroy_board_pad();
    return;
  }

  int inner_width = board_width - 2;
  int desired_height = board_height + BOARD_PAD_BUFFER_ROWS;
  if (desired_height < board_height) {
    desired_height = board_height;
  }
  if (desired_height > BOARD_PAD_MAX_HEIGHT) {
    desired_height = BOARD_PAD_MAX_HEIGHT;
  }
  int desired_width = inner_width + 2;

  if (board_pad && board_pad_height == desired_height && board_pad_width == desired_width) {
    return;
  }

  destroy_board_pad();
  board_pad = newpad(desired_height, desired_width);
  if (!board_pad) {
    return;
  }
  board_pad_height = desired_height;
  board_pad_width = desired_width;
}

static void clamp_board_scroll(void) {
  if (!board_win) {
    board_scroll_offset = 0;
    return;
  }

  int visible_inner_height = getmaxy(board_win) - 2;
  if (visible_inner_height <= 0) {
    board_scroll_offset = 0;
    return;
  }

  int max_scroll = board_content_height - visible_inner_height;
  if (max_scroll < 0) {
    max_scroll = 0;
  }
  if (board_scroll_offset > max_scroll) {
    board_scroll_offset = max_scroll;
  }
  if (board_scroll_offset < 0) {
    board_scroll_offset = 0;
  }
}

static void refresh_board_pad_view(void) {
  if (!board_win || !board_pad) {
    return;
  }
  int inner_height = getmaxy(board_win) - 2;
  int inner_width = getmaxx(board_win) - 2;
  if (inner_height <= 0 || inner_width <= 0) {
    return;
  }

  clamp_board_scroll();

  int content_max_row = board_pad_height - 2;
  int start_row = board_scroll_offset + 1;
  if (start_row > content_max_row) {
    start_row = content_max_row;
  }
  if (start_row < 1) {
    start_row = 1;
  }
  int end_row = start_row + inner_height - 1;
  if (end_row > content_max_row) {
    end_row = content_max_row;
    start_row = end_row - inner_height + 1;
    if (start_row < 1) {
      start_row = 1;
    }
  }

  int board_begy = 0;
  int board_begx = 0;
  getbegyx(board_win, board_begy, board_begx);

  pnoutrefresh(board_pad,
               start_row,
               1,
               board_begy + 1,
               board_begx + 1,
               board_begy + inner_height,
               board_begx + inner_width);
  doupdate();
}

static void scroll_board_to_offset(int offset) {
  if (!board_win || !board_pad) {
    return;
  }

  clamp_board_scroll();
  int visible_inner_height = getmaxy(board_win) - 2;
  int max_scroll = board_content_height - visible_inner_height;
  if (max_scroll < 0) {
    max_scroll = 0;
  }
  if (offset < 0) {
    offset = 0;
  }
  if (offset > max_scroll) {
    offset = max_scroll;
  }

  if (offset == board_scroll_offset) {
    return;
  }

  board_scroll_offset = offset;
  refresh_board_pad_view();
}

static void scroll_board_lines(int delta) {
  if (delta == 0 || !board_win || !board_pad) {
    return;
  }
  scroll_board_to_offset(board_scroll_offset + delta);
}

static void scroll_board_page(int direction) {
  if (!board_win || direction == 0) {
    return;
  }
  int inner_height = getmaxy(board_win) - 2;
  if (inner_height <= 0) {
    return;
  }
  scroll_board_to_offset(board_scroll_offset + direction * inner_height);
}

static void destroy_windows(void) {
  destroy_board_pad();
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

static char cooldown_state_char(const TapState *tap_state) {
  if (!tap_state) {
    return '-';
  }
  return tap_state->cooldown ? 'C' : 'R';
}

static size_t card_observation_count(const CardObservationData *cards, size_t max_count) {
  if (!cards) {
    return 0;
  }
  size_t count = 0;
  for (size_t i = 0; i < max_count; i++) {
    if (cards[i].id.code) {
      count++;
    }
  }
  return count;
}

static size_t ikz_observation_count(const IKZCardObservationData *cards, size_t max_count) {
  if (!cards) {
    return 0;
  }
  size_t count = 0;
  for (size_t i = 0; i < max_count; i++) {
    if (cards[i].id.code) {
      count++;
    }
  }
  return count;
}

static const char *card_type_to_string(CardType type) {
  switch (type) {
    case CARD_TYPE_LEADER:
      return "Leader";
    case CARD_TYPE_GATE:
      return "Gate";
    case CARD_TYPE_ENTITY:
      return "Entity";
    case CARD_TYPE_WEAPON:
      return "Weapon";
    case CARD_TYPE_SPELL:
      return "Spell";
    case CARD_TYPE_IKZ:
      return "IKZ";
    case CARD_TYPE_EXTRA_IKZ:
      return "Extra IKZ";
    default:
      return "Unknown";
  }
}

static void draw_text_box(WINDOW *win, int top, int left, int width, int height, const char *lines[], size_t line_count) {
  if (!win || width < 3 || height < 3) {
    return;
  }

  int max_y = getmaxy(win);
  int max_x = getmaxx(win);
  if (top < 1 || left < 1 || top + height > max_y - 1 || left + width > max_x - 1) {
    return;
  }

  mvwaddch(win, top, left, ACS_ULCORNER);
  mvwaddch(win, top, left + width - 1, ACS_URCORNER);
  mvwaddch(win, top + height - 1, left, ACS_LLCORNER);
  mvwaddch(win, top + height - 1, left + width - 1, ACS_LRCORNER);
  mvwhline(win, top, left + 1, ACS_HLINE, width - 2);
  mvwhline(win, top + height - 1, left + 1, ACS_HLINE, width - 2);
  mvwvline(win, top + 1, left, ACS_VLINE, height - 2);
  mvwvline(win, top + 1, left + width - 1, ACS_VLINE, height - 2);

  for (int y = top + 1; y < top + height - 1; ++y) {
    for (int x = left + 1; x < left + width - 1; ++x) {
      mvwaddch(win, y, x, ' ');
    }
  }

  int inner_width = width - 2;
  int max_lines = height - 2;
  for (int i = 0; i < max_lines && (size_t)i < line_count; ++i) {
    const char *line = lines[i];
    if (!line || line[0] == '\0') {
      continue;
    }
    mvwprintw(win, top + 1 + i, left + 1, "%.*s", inner_width, line);
  }
}

static void build_standard_card_lines(const CardObservationData *card, char lines[4][CARD_BOX_TEXT_CAPACITY]) {
  const char *code = card_code_or_placeholder(&card->id);
  snprintf(lines[0], CARD_BOX_TEXT_CAPACITY, "%s", code);

  const char *type_str = card_type_to_string(card->type.value);
  snprintf(lines[1], CARD_BOX_TEXT_CAPACITY, "Type: %s", type_str);

  if (card->has_cur_stats) {
    snprintf(lines[2], CARD_BOX_TEXT_CAPACITY, "ATK:%d HP:%d", card->cur_stats.cur_atk, card->cur_stats.cur_hp);
  } else if (card->has_gate_points) {
    snprintf(lines[2], CARD_BOX_TEXT_CAPACITY, "GP: %u", card->gate_points.gate_points);
  } else if (card->ikz_cost.ikz_cost) {
    snprintf(lines[2], CARD_BOX_TEXT_CAPACITY, "IKZ Cost: %d", card->ikz_cost.ikz_cost);
  } else {
    snprintf(lines[2], CARD_BOX_TEXT_CAPACITY, "Stats: --");
  }

  char tap = tap_state_char(&card->tap_state);
  char cooldown = cooldown_state_char(&card->tap_state);
  if (card->ikz_cost.ikz_cost) {
    snprintf(lines[3], CARD_BOX_TEXT_CAPACITY, "T/C:%c/%c IKZ:%d", tap, cooldown, card->ikz_cost.ikz_cost);
  } else {
    snprintf(lines[3], CARD_BOX_TEXT_CAPACITY, "T/C:%c/%c", tap, cooldown);
  }
}

static void draw_standard_card_box(WINDOW *win, int top, int left, const CardObservationData *card) {
  if (!win || !card) {
    return;
  }
  char lines[4][CARD_BOX_TEXT_CAPACITY];
  memset(lines, 0, sizeof lines);
  build_standard_card_lines(card, lines);
  const char *line_ptrs[4] = { lines[0], lines[1], lines[2], lines[3] };
  draw_text_box(win, top, left, CARD_BOX_WIDTH, CARD_BOX_HEIGHT, line_ptrs, 4);
}

static void draw_empty_card_box(WINDOW *win, int top, int left) {
  if (!win) {
    return;
  }
  const char *no_lines[1] = { NULL };
  draw_text_box(win, top, left, CARD_BOX_WIDTH, CARD_BOX_HEIGHT, no_lines, 0);
}

static void draw_leader_box(WINDOW *win, int top, int left, const LeaderCardObservationData *leader) {
  if (!win || !leader) {
    return;
  }
  CardObservationData as_card = {0};
  as_card.type = leader->type;
  as_card.id = leader->id;
  as_card.tap_state = leader->tap_state;
  as_card.cur_stats = leader->cur_stats;
  as_card.has_cur_stats = true;
  as_card.has_gate_points = false;
  as_card.ikz_cost.ikz_cost = 0;
  draw_standard_card_box(win, top, left, &as_card);
}

static void draw_gate_box(WINDOW *win, int top, int left, const GateCardObservationData *gate) {
  if (!win || !gate) {
    return;
  }
  CardObservationData as_card = {0};
  as_card.type = gate->type;
  as_card.id = gate->id;
  as_card.tap_state = gate->tap_state;
  as_card.has_cur_stats = false;
  as_card.has_gate_points = false;
  as_card.ikz_cost.ikz_cost = 0;
  draw_standard_card_box(win, top, left, &as_card);
}

static void draw_ikz_card_box(WINDOW *win, int top, int left, const IKZCardObservationData *card) {
  if (!win || !card) {
    return;
  }
  char lines[4][CARD_BOX_TEXT_CAPACITY];
  memset(lines, 0, sizeof lines);
  const char *code = card_code_or_placeholder(&card->id);
  snprintf(lines[0], CARD_BOX_TEXT_CAPACITY, "%s", code);
  const char *type_str = card_type_to_string(card->type.value);
  snprintf(lines[1], CARD_BOX_TEXT_CAPACITY, "Type: %s", type_str);
  char tap = tap_state_char(&card->tap_state);
  char cooldown = cooldown_state_char(&card->tap_state);
  snprintf(lines[2], CARD_BOX_TEXT_CAPACITY, "T/C:%c/%c", tap, cooldown);
  const char *line_ptrs[4] = { lines[0], lines[1], lines[2], lines[3] };
  draw_text_box(win, top, left, CARD_BOX_WIDTH, CARD_BOX_HEIGHT, line_ptrs, 4);
}

static int compute_card_columns(WINDOW *win) {
  int available_width = getmaxx(win) - 4;
  int per_card = CARD_BOX_WIDTH + 1;
  int cols = (available_width + 1) / per_card;
  if (cols < 1) {
    cols = 1;
  }
  return cols;
}

static int draw_leader_gate_section(WINDOW *win, int row, int max_inner_row,
                                    const LeaderCardObservationData *leader,
                                    const GateCardObservationData *gate) {
  if (!win) {
    return row;
  }
  if (row > max_inner_row) {
    return row;
  }
  mvwprintw(win, row, 2, "Leader & Gate");
  row++;
  if (row > max_inner_row) {
    return row;
  }

  int top = row;
  if (top + CARD_BOX_HEIGHT - 1 > max_inner_row) {
    mvwprintw(win, max_inner_row, 4, "(not enough space)");
    return max_inner_row + 1;
  }

  draw_leader_box(win, top, 2, leader);
  int gate_left = 2 + CARD_BOX_WIDTH + 1;
  int max_x = getmaxx(win);
  if (gate_left + CARD_BOX_WIDTH <= max_x - 1) {
    draw_gate_box(win, top, gate_left, gate);
    row = top + CARD_BOX_HEIGHT;
  } else {
    row = top + CARD_BOX_HEIGHT;
    if (row <= max_inner_row) {
      row++;
    }
    if (row + CARD_BOX_HEIGHT - 1 > max_inner_row) {
      mvwprintw(win, max_inner_row, 4, "(not enough space)");
      return max_inner_row + 1;
    }
    draw_gate_box(win, row, 2, gate);
    row += CARD_BOX_HEIGHT;
  }

  if (row <= max_inner_row) {
    row++;
  }
  return row;
}

static int draw_card_grid_section(WINDOW *win, int row, int max_inner_row, const char *label,
                                  const CardObservationData *cards, size_t max_count,
                                  bool use_zone_index) {
  if (!win) {
    return row;
  }
  if (row > max_inner_row) {
    return row;
  }
  mvwprintw(win, row, 2, "%s", label);
  row++;
  if (row > max_inner_row) {
    return row;
  }

  size_t count = card_observation_count(cards, max_count);

  if (!use_zone_index && count == 0) {
    mvwprintw(win, row, 4, "(empty)");
    row++;
  } else {
    int cols = compute_card_columns(win);
    size_t total_slots = use_zone_index ? max_count : count;
    size_t index = 0;
    CardObservationData ordered_cards[max_count];
    bool slot_has_card[max_count];
    if (use_zone_index) {
      for (size_t i = 0; i < max_count; ++i) {
        ordered_cards[i] = (CardObservationData){0};
        slot_has_card[i] = false;
      }

      size_t fallback_slot = 0;
      for (size_t i = 0; i < max_count; ++i) {
        const CardObservationData *card = &cards[i];
        if (!card->id.code) {
          continue;
        }
        size_t target_index = max_count;
        if (card->has_zone_index && card->zone_index < max_count && !slot_has_card[card->zone_index]) {
          target_index = card->zone_index;
        }
        if (target_index == max_count) {
          while (fallback_slot < max_count && slot_has_card[fallback_slot]) {
            fallback_slot++;
          }
          if (fallback_slot < max_count) {
            target_index = fallback_slot;
            fallback_slot++;
          }
        }
        if (target_index < max_count) {
          ordered_cards[target_index] = *card;
          slot_has_card[target_index] = true;
        }
      }
    }

    while (index < total_slots) {
      int top = row;
      if (top + CARD_BOX_HEIGHT - 1 > max_inner_row) {
        mvwprintw(win, max_inner_row, 4, "(truncated)");
        return max_inner_row + 1;
      }
      int drawn_this_row = 0;
      for (int col = 0; col < cols && index < total_slots; ++col) {
        int left = 2 + col * (CARD_BOX_WIDTH + 1);
        if (left + CARD_BOX_WIDTH > getmaxx(win) - 1) {
          break;
        }
        if (use_zone_index) {
          if (slot_has_card[index]) {
            draw_standard_card_box(win, top, left, &ordered_cards[index]);
          } else {
            draw_empty_card_box(win, top, left);
          }
        } else {
          draw_standard_card_box(win, top, left, &cards[index]);
        }
        index++;
        drawn_this_row++;
      }
      if (drawn_this_row == 0) {
        mvwprintw(win, row, 4, "(not enough width)");
        row++;
        break;
      }
      row += CARD_BOX_HEIGHT;
      if (index < total_slots) {
        row++;
      }
    }
  }

  if (row <= max_inner_row) {
    row++;
  }
  return row;
}

static int draw_ikz_grid_section(WINDOW *win, int row, int max_inner_row, const char *label,
                                 const IKZCardObservationData *cards, size_t count) {
  if (!win) {
    return row;
  }
  if (row > max_inner_row) {
    return row;
  }
  mvwprintw(win, row, 2, "%s", label);
  row++;
  if (row > max_inner_row) {
    return row;
  }

  if (count == 0) {
    mvwprintw(win, row, 4, "(empty)");
    row++;
  } else {
    int cols = compute_card_columns(win);
    size_t index = 0;
    while (index < count) {
      int top = row;
      if (top + CARD_BOX_HEIGHT - 1 > max_inner_row) {
        mvwprintw(win, max_inner_row, 4, "(truncated)");
        return max_inner_row + 1;
      }
      int drawn_this_row = 0;
      for (int col = 0; col < cols && index < count; ++col) {
        int left = 2 + col * (CARD_BOX_WIDTH + 1);
        if (left + CARD_BOX_WIDTH > getmaxx(win) - 1) {
          break;
        }
        draw_ikz_card_box(win, top, left, &cards[index]);
        index++;
        drawn_this_row++;
      }
      if (drawn_this_row == 0) {
        mvwprintw(win, row, 4, "(not enough width)");
        row++;
        break;
      }
      row += CARD_BOX_HEIGHT;
      if (index < count) {
        row++;
      }
    }
  }

  if (row <= max_inner_row) {
    row++;
  }
  return row;
}

static int draw_opponent_info_section(WINDOW *win, int row, int max_inner_row,
                                      const OpponentObservationData *data) {
  if (!win || !data) {
    return row;
  }
  if (row > max_inner_row) {
    return row;
  }
  mvwprintw(win, row, 2, "Info:");
  row++;
  if (row > max_inner_row) {
    return row;
  }

  mvwprintw(win, row++, 4, "Hand: %u  IKZ Pile: %u  Discard: %u",
            data->hand_count,
            data->ikz_pile_count,
            data->discard_count);
  if (row > max_inner_row) {
    return row;
  }
  mvwprintw(win, row++, 4, "IKZ Token: %s", data->has_ikz_token ? "Yes" : "No");
  if (row <= max_inner_row) {
    row++;
  }
  return row;
}

static int draw_my_info_section(WINDOW *win, int row, int max_inner_row,
                                const MyObservationData *data, size_t hand_count) {
  if (!win || !data) {
    return row;
  }
  if (row > max_inner_row) {
    return row;
  }
  mvwprintw(win, row, 2, "Info:");
  row++;
  if (row > max_inner_row) {
    return row;
  }

  mvwprintw(win, row++, 4, "Hand: %zu  IKZ Pile: %u  Discard: %u",
            hand_count,
            data->ikz_pile_count,
            data->discard_count);
  if (row > max_inner_row) {
    return row;
  }
  mvwprintw(win, row++, 4, "IKZ Token: %s", data->has_ikz_token ? "Yes" : "No");
  if (row <= max_inner_row) {
    row++;
  }
  return row;
}

static int render_opponent_board_column(WINDOW *win, const OpponentObservationData *opponent) {
  if (!win || !opponent) {
    return 0;
  }

  werase(win);

  int max_inner_row = getmaxy(win) - 1;
  if (max_inner_row <= 0) {
    return 0;
  }

  int row = 0;
  mvwprintw(win, row++, 1, "Opponent Board");
  if (row > max_inner_row) {
    return row;
  }

  row = draw_leader_gate_section(win, row, max_inner_row, &opponent->leader, &opponent->gate);
  if (row > max_inner_row) {
    return row;
  }

  row = draw_card_grid_section(win, row, max_inner_row, "Garden", opponent->garden, GARDEN_SIZE, true);
  if (row > max_inner_row) {
    return row;
  }

  row = draw_card_grid_section(win, row, max_inner_row, "Alley", opponent->alley, ALLEY_SIZE, true);
  if (row > max_inner_row) {
    return row;
  }

  size_t opp_ikz_count = ikz_observation_count(opponent->ikz_area, IKZ_AREA_SIZE);
  row = draw_ikz_grid_section(win, row, max_inner_row, "IKZ Area", opponent->ikz_area, opp_ikz_count);
  if (row > max_inner_row) {
    return row;
  }

  row = draw_opponent_info_section(win, row, max_inner_row, opponent);
  return row;
}

static int render_my_board_column(WINDOW *win, const MyObservationData *mine) {
  if (!win || !mine) {
    return 0;
  }

  werase(win);

  int max_inner_row = getmaxy(win) - 1;
  if (max_inner_row <= 0) {
    return 0;
  }

  int row = 0;
  mvwprintw(win, row++, 1, "Your Board");
  if (row > max_inner_row) {
    return row;
  }

  row = draw_leader_gate_section(win, row, max_inner_row, &mine->leader, &mine->gate);
  if (row > max_inner_row) {
    return row;
  }

  row = draw_card_grid_section(win, row, max_inner_row, "Garden", mine->garden, GARDEN_SIZE, true);
  if (row > max_inner_row) {
    return row;
  }

  row = draw_card_grid_section(win, row, max_inner_row, "Alley", mine->alley, ALLEY_SIZE, true);
  if (row > max_inner_row) {
    return row;
  }

  size_t my_hand_count = card_observation_count(mine->hand, MAX_HAND_SIZE);
  row = draw_card_grid_section(win, row, max_inner_row, "Hand", mine->hand, MAX_HAND_SIZE, false);
  if (row > max_inner_row) {
    return row;
  }

  size_t my_ikz_count = ikz_observation_count(mine->ikz_area, IKZ_AREA_SIZE);
  row = draw_ikz_grid_section(win, row, max_inner_row, "IKZ Area", mine->ikz_area, my_ikz_count);
  if (row > max_inner_row) {
    return row;
  }

  row = draw_my_info_section(win, row, max_inner_row, mine, my_hand_count);
  return row;
}

static int render_board_single_column(WINDOW *win, const ObservationData *observation, const GameState *gs) {
  if (!win || !observation || !gs) {
    return 0;
  }

  const int max_inner_row = getmaxy(win) - 2;
  if (max_inner_row <= 0) {
    return 0;
  }

  int row = 1;
  const OpponentObservationData *opponent = &observation->opponent_observation_data;
  const MyObservationData *mine = &observation->my_observation_data;

  mvwprintw(win, row++, 2, "Opponent Board");
  if (row > max_inner_row) {
    return row - 1;
  }

  row = draw_leader_gate_section(win, row, max_inner_row, &opponent->leader, &opponent->gate);
  if (row > max_inner_row) {
    return row - 1;
  }

  row = draw_card_grid_section(win, row, max_inner_row, "Garden", opponent->garden, GARDEN_SIZE, true);
  if (row > max_inner_row) {
    return row - 1;
  }

  row = draw_card_grid_section(win, row, max_inner_row, "Alley", opponent->alley, ALLEY_SIZE, true);
  if (row > max_inner_row) {
    return row - 1;
  }

  size_t opp_ikz_count = ikz_observation_count(opponent->ikz_area, IKZ_AREA_SIZE);
  row = draw_ikz_grid_section(win, row, max_inner_row, "IKZ Area", opponent->ikz_area, opp_ikz_count);
  if (row > max_inner_row) {
    return row - 1;
  }

  row = draw_opponent_info_section(win, row, max_inner_row, opponent);
  if (row > max_inner_row) {
    return row - 1;
  }

  if (row <= max_inner_row) {
    row++;
  }
  if (row > max_inner_row) {
    return row - 1;
  }

  mvwprintw(win, row++, 2, "Your Board");
  if (row > max_inner_row) {
    return row - 1;
  }

  row = draw_leader_gate_section(win, row, max_inner_row, &mine->leader, &mine->gate);
  if (row > max_inner_row) {
    return row - 1;
  }

  row = draw_card_grid_section(win, row, max_inner_row, "Garden", mine->garden, GARDEN_SIZE, true);
  if (row > max_inner_row) {
    return row - 1;
  }

  row = draw_card_grid_section(win, row, max_inner_row, "Alley", mine->alley, ALLEY_SIZE, true);
  if (row > max_inner_row) {
    return row - 1;
  }

  size_t my_hand_count = card_observation_count(mine->hand, MAX_HAND_SIZE);
  row = draw_card_grid_section(win, row, max_inner_row, "Hand", mine->hand, MAX_HAND_SIZE, false);
  if (row > max_inner_row) {
    return row - 1;
  }

  size_t my_ikz_count = ikz_observation_count(mine->ikz_area, IKZ_AREA_SIZE);
  row = draw_ikz_grid_section(win, row, max_inner_row, "IKZ Area", mine->ikz_area, my_ikz_count);
  if (row > max_inner_row) {
    return row - 1;
  }

  row = draw_my_info_section(win, row, max_inner_row, mine, my_hand_count);
  return row > 1 ? row - 1 : 0;
}

static int render_board(WINDOW *win, const ObservationData *observation, const GameState *gs) {
  if (!win || !observation || !gs) {
    return 0;
  }

  const int inner_height = getmaxy(win) - 2;
  const int inner_width = getmaxx(win) - 2;
  if (inner_height <= 0 || inner_width <= 0) {
    return 0;
  }

  const int min_column_width = CARD_BOX_WIDTH + 4;
  if (inner_width < min_column_width * 2) {
    return render_board_single_column(win, observation, gs);
  }

  int left_width = inner_width / 2;
  int right_width = inner_width - left_width;
  if (left_width < min_column_width || right_width < min_column_width) {
    return render_board_single_column(win, observation, gs);
  }

  WINDOW *left_win = derwin(win, inner_height, left_width, 1, 1);
  WINDOW *right_win = derwin(win, inner_height, right_width, 1, 1 + left_width);
  if (!left_win || !right_win) {
    if (left_win) {
      delwin(left_win);
    }
    if (right_win) {
      delwin(right_win);
    }
    return render_board_single_column(win, observation, gs);
  }

  int left_height = render_my_board_column(left_win, &observation->my_observation_data);
  int right_height = render_opponent_board_column(right_win, &observation->opponent_observation_data);

  touchwin(win);

  delwin(left_win);
  delwin(right_win);

  return (left_height > right_height) ? left_height : right_height;
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

  ensure_board_pad();

  last_observation = *observation;
  last_state = *gs;
  has_last_state = true;

  werase(board_win);
  box(board_win, 0, 0);
  if (board_pad) {
    werase(board_pad);
    board_content_height = render_board(board_pad, observation, gs);
    if (board_content_height < 0) {
      board_content_height = 0;
    }
    if (board_content_height > board_pad_height - 2) {
      board_content_height = board_pad_height - 2;
    }
    clamp_board_scroll();
    wrefresh(board_win);
    refresh_board_pad_view();
  } else {
    board_content_height = render_board(board_win, observation, gs);
    board_scroll_offset = 0;
    wrefresh(board_win);
  }

  refresh_log_window();

  werase(info_win);
  box(info_win, 0, 0);
  render_info(info_win, gs);
  wrefresh(info_win);

  werase(input_win);
  box(input_win, 0, 0);
  mvwprintw(input_win, 1, 2, "No input required.");
  if (getmaxy(input_win) >= 4) {
    mvwprintw(input_win, 2, 2, "Scroll board with Up/Down or PgUp/PgDn");
  }
  wrefresh(input_win);

  curs_set(0);
}

bool cli_render_prompt_user_action(int active_player_index, const char *message, char *buffer, size_t buffer_length) {
  if (!buffer || buffer_length == 0) {
    return false;
  }

  buffer[0] = '\0';
  size_t input_length = 0;
  bool success = false;

  ensure_windows();

  if (!input_win) {
    return false;
  }

  curs_set(1);

  while (true) {
    ensure_windows();
    if (!input_win) {
      break;
    }
    refresh_log_window();

    werase(input_win);
    box(input_win, 0, 0);

    int max_inner_rows = getmaxy(input_win) - 2;
    if (max_inner_rows < 1) {
      max_inner_rows = 1;
    }
    int current_row = 1;
    if (message && message[0] != '\0' && current_row <= max_inner_rows) {
      mvwprintw(input_win, current_row++, 2, "%s", message);
    }
    if (current_row <= max_inner_rows) {
      mvwprintw(input_win, current_row++, 2, "Awaiting user action [Player %d] (type,p0,p1,p2):", active_player_index);
    }
    if (current_row <= max_inner_rows) {
      mvwprintw(input_win, current_row++, 2, "Scroll board with Up/Down or PgUp/PgDn");
    }
    if (current_row > max_inner_rows) {
      current_row = max_inner_rows;
    }
    if (current_row < 1) {
      current_row = 1;
    }
    wmove(input_win, current_row, 2);
    wclrtoeol(input_win);
    if (input_length > 0) {
      waddnstr(input_win, buffer, (int)input_length);
    }
    wmove(input_win, current_row, 2 + (int)input_length);
    wrefresh(input_win);

    int ch = wgetch(input_win);
    if (ch == ERR) {
      if (is_term_resized(screen_rows, screen_cols)) {
        if (has_last_state) {
          cli_render_draw(&last_observation, &last_state);
        } else {
          refresh();
        }
        continue;
      }
      break;
    }

    if (ch == KEY_RESIZE) {
      if (has_last_state) {
        cli_render_draw(&last_observation, &last_state);
      } else {
        refresh();
      }
      continue;
    }

    switch (ch) {
      case KEY_PPAGE:
        scroll_board_page(-1);
        continue;
      case KEY_NPAGE:
        scroll_board_page(1);
        continue;
      case KEY_UP:
        scroll_board_lines(-1);
        continue;
      case KEY_DOWN:
        scroll_board_lines(1);
        continue;
      case KEY_HOME:
        scroll_board_to_offset(0);
        continue;
      case KEY_END:
        scroll_board_to_offset(board_content_height);
        continue;
      case KEY_BACKSPACE:
      case 127:
      case 8:
        if (input_length > 0) {
          input_length--;
          buffer[input_length] = '\0';
        }
        continue;
      case 21: // Ctrl+U
        input_length = 0;
        buffer[0] = '\0';
        continue;
      case KEY_ENTER:
      case '\n':
      case '\r':
        buffer[input_length] = '\0';
        success = true;
        goto exit_loop;
      default:
        break;
    }

    if (ch >= 32 && ch <= 126) {
      if (input_length < buffer_length - 1) {
        buffer[input_length++] = (char)ch;
        buffer[input_length] = '\0';
      }
    }
  }

exit_loop:
  curs_set(0);
  return success;
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
