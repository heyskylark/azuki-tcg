#ifndef AZUKI_UTILS_DEBUG_LOG_H
#define AZUKI_UTILS_DEBUG_LOG_H

#include <stdbool.h>
#include <stdint.h>
#include <stdarg.h>

#define AZK_DEBUG_LOG_MAX_ENTRIES 256
#define AZK_DEBUG_LOG_MAX_LENGTH 256

typedef enum {
  AZK_DEBUG_LEVEL_INFO,
  AZK_DEBUG_LEVEL_WARN,
  AZK_DEBUG_LEVEL_ERROR
} AzkDebugLevel;

typedef struct {
  AzkDebugLevel level;
  char message[AZK_DEBUG_LOG_MAX_LENGTH];
} AzkDebugLogEntry;

/**
 * Initialize the debug log system.
 * Called automatically on first use, but can be called explicitly.
 */
void azk_debug_log_init(void);

/**
 * Enable or disable debug logging.
 * Disabled by default for performance.
 */
void azk_debug_log_set_enabled(bool enabled);

/**
 * Check if debug logging is enabled.
 */
bool azk_debug_log_is_enabled(void);

/**
 * Clear all debug log entries.
 */
void azk_debug_log_clear(void);

/**
 * Log a formatted debug message.
 * Only logs if debug logging is enabled.
 */
void azk_debug_logf(AzkDebugLevel level, const char *fmt, ...);

/**
 * Get the current number of log entries.
 */
uint16_t azk_debug_log_count(void);

/**
 * Get all log entries.
 * Returns pointer to internal buffer and sets out_count to number of entries.
 */
const AzkDebugLogEntry *azk_debug_log_get_entries(uint16_t *out_count);

/* Convenience macros for logging at different levels */
#define AZK_DEBUG_INFO(fmt, ...) azk_debug_logf(AZK_DEBUG_LEVEL_INFO, fmt, ##__VA_ARGS__)
#define AZK_DEBUG_WARN(fmt, ...) azk_debug_logf(AZK_DEBUG_LEVEL_WARN, fmt, ##__VA_ARGS__)
#define AZK_DEBUG_ERROR(fmt, ...) azk_debug_logf(AZK_DEBUG_LEVEL_ERROR, fmt, ##__VA_ARGS__)

#endif /* AZUKI_UTILS_DEBUG_LOG_H */
