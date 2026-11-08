// A diagnostic for a GPU that gives wrong results or fails, run on the phone
// itself. With KITTENTTS2_CHECK set to a file path, every graph runs one node at a
// time, and in the first graph of each size every node the GPU ran is run again
// on the CPU from the same inputs. The file gets:
//   - the nodes whose results differ from the CPU's,
//   - inputs that changed between the node that made them and the node that
//     reads them (a shader writing past its own tensor),
//   - the node that was running when the GPU failed,
// and `<file>.ops` gets the time spent in each op, on the GPU and on the CPU.
//
// Nothing changes when KITTENTTS2_CHECK is unset. It is slow when set: a short
// sentence only.

#pragma once

#include "ggml-backend.h"

namespace kt2 {

/**
 * Computes `graph` on `sched` with the checks above and returns true, or returns
 * false without computing when KITTENTTS2_CHECK is unset. Throws when the
 * computation fails.
 */
bool computeChecked(ggml_backend_sched_t sched, ggml_cgraph * graph);

} // namespace kt2
