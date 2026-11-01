/* slow_recon.c — a RECONSTRUCTION of the timing symptom that exposed the 2024 xz backdoor.
 *
 * This is NOT the xz backdoor and contains NO part of it. It does exactly one thing: when it is loaded
 * into a process, a constructor burns a bounded amount of CPU and returns. That reproduces the *symptom*
 * Andres Freund measured — an sshd login that suddenly took about half a second longer — without any of
 * the mechanism: no cryptography is hooked, no authentication is bypassed, no network traffic is sent,
 * no command is run. It is loaded with LD_PRELOAD, in the open, so nothing here is hidden either.
 *
 * The real backdoor spent its extra time parsing symbol tables in memory to find the function it wanted
 * to hijack. We imitate only the SHAPE of that cost with a plain arithmetic loop over a fake in-memory
 * table, calibrated to a target number of milliseconds.
 *
 *   Gate:  the burn runs only when RECON_ACTIVE=1, and is skipped when RECON_QUIET=1 dials it down.
 *          (The real one gated itself on how the process was launched; we gate on an env var so the same
 *           binary gives both the baseline and the slow path.)
 *
 *   cc -O2 -fPIC -shared slow_recon.c -o libslow-recon.so
 */
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <string.h>

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

/* A bounded busy-loop that mimics "parse a symbol table" work: hash a fake table until the target
 * wall-time is reached. Volatile sink so the optimiser cannot delete it. */
static volatile uint64_t sink;

__attribute__((constructor))
static void recon_startup(void) {
    if (!getenv("RECON_ACTIVE")) return;                 /* baseline: do nothing */
    double target = 500.0;                               /* default: ~500 ms, Freund's ~0.3->0.8s gap */
    const char *ms = getenv("RECON_MS");
    if (ms) target = atof(ms);
    if (getenv("RECON_QUIET")) target *= 0.15;           /* a stealthier version: narrow the gap */

    /* a fake "symbol table" to chew through, so the work resembles table parsing, not just spinning */
    enum { N = 4096 };
    static uint64_t table[N];
    for (int i = 0; i < N; i++) table[i] = (uint64_t)i * 2654435761u;

    double start = now_ms();
    uint64_t h = 1469598103934665603ull;                 /* FNV-ish rolling hash */
    while (now_ms() - start < target) {
        for (int i = 0; i < N; i++) {
            h ^= table[i];
            h *= 1099511628211ull;
            table[i] = h ^ (h >> 17);
        }
    }
    sink = h;
}
