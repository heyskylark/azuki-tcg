#include "utils/debug_log.h"

#include <stdio.h>
#include <string.h>

/* Ring buffer for debug log entries */
static AzkDebugLogEntry debug_log_buffer[AZK_DEBUG_LOG_MAX_ENTRIES];
static uint16_t debug_log_head = 0;
static uint16_t debug_log_count = 0;
static bool debug_log_enabled = false;
static bool debug_log_initialized = false;

void azk_debug_log_init(void) {
  if (debug_log_initialized) {
    return;
  }
  memset(debug_log_buffer, 0, sizeof(debug_log_buffer));
  debug_log_head = 0;
  debug_log_count = 0;
  debug_log_enabled = false;
  debug_log_initialized = true;
}

void azk_debug_log_set_enabled(bool enabled) {
  if (!debug_log_initialized) {
    azk_debug_log_init();
  }
  debug_log_enabled = enabled;
}

bool azk_debug_log_is_enabled(void) {
  return debug_log_enabled;
}

void azk_debug_log_clear(void) {
  debug_log_head = 0;
  debug_log_count = 0;
}

void azk_debug_logf(AzkDebugLevel level, const char *fmt, ...) {
  if (!debug_log_enabled) {
    return;
  }

  if (!debug_log_initialized) {
    azk_debug_log_init();
  }

  /* Get the next slot in the ring buffer */
  AzkDebugLogEntry *entry = &debug_log_buffer[debug_log_head];
  entry->level = level;

  /* Format the message */
  va_list args;
  va_start(args, fmt);
  vsnprintf(entry->message, AZK_DEBUG_LOG_MAX_LENGTH, fmt, args);
  va_end(args);

  /* Ensure null termination */
  entry->message[AZK_DEBUG_LOG_MAX_LENGTH - 1] = '\0';

  /* Advance head pointer (ring buffer) */
  debug_log_head = (debug_log_head + 1) % AZK_DEBUG_LOG_MAX_ENTRIES;

  /* Update count (capped at max) */
  if (debug_log_count < AZK_DEBUG_LOG_MAX_ENTRIES) {
    debug_log_count++;
  }
}

uint16_t azk_debug_log_count(void) {
  return debug_log_count;
}

const AzkDebugLogEntry *azk_debug_log_get_entries(uint16_t *out_count) {
  if (out_count) {
    *out_count = debug_log_count;
  }

  /* If buffer hasn't wrapped, entries start at 0 */
  if (debug_log_count < AZK_DEBUG_LOG_MAX_ENTRIES) {
    return debug_log_buffer;
  }

  /* Buffer has wrapped - entries start at head (oldest entry) */
  /* For simplicity, we return from head to end, then 0 to head-1 */
  /* Caller should iterate: for i in 0..count: buffer[(head + i) % max] */
  /* But for linear access, we just return the buffer and let caller handle it */
  return debug_log_buffer;
}
