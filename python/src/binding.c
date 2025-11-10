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
    assign_to_dict(dict, "p1_winrate", log->p1_winrate);
    assign_to_dict(dict, "p2_winrate", log->p2_winrate);
    assign_to_dict(dict, "draw_rate", log->draw_rate);
    assign_to_dict(dict, "n", log->n);
    return 0;
}

