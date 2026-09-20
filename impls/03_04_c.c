/*
 * 03 / 04 — the same C, twice.
 *
 * ONE source file, two builds. The ONLY difference is the body of matmul():
 *
 *   -DUSE_ACCELERATE=0  -> three nested loops I wrote            (impl 03-c-naive)
 *   -DUSE_ACCELERATE=1  -> one call to cblas_sgemm               (impl 04-c-accelerate)
 *
 * Everything else — the loading, the attention, the layernorm, the gelu, the argmax — is
 * byte-identical between the two binaries. That is deliberate: if they post different times,
 * the matmul is the only thing that could have caused it.
 *
 * Build:  see harness/build_c.sh
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#if USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#define IMPL_NAME "04-c-accelerate"
#else
#define IMPL_NAME "03-c-naive"
#endif

#define MAXT 256

static int d_model, n_head, n_layer, d_ff, block_size, vocab_size, new_tokens, total_floats;
static int head_dim;
static float *W; /* the whole weight blob */

typedef struct { char name[64]; int offset, count; } Tensor;
static Tensor tensors[512];
static int n_tensors = 0;

static int prompt[MAXT], prompt_len;
static int vocab_cp[512];

static void die(const char *m) { fprintf(stderr, "fatal: %s\n", m); exit(1); }

static float *T(const char *name) {
  for (int i = 0; i < n_tensors; i++)
    if (!strcmp(tensors[i].name, name)) return W + tensors[i].offset;
  fprintf(stderr, "missing tensor %s\n", name);
  exit(1);
}

static void load_layout(const char *path) {
  FILE *f = fopen(path, "r");
  if (!f) die("cannot open layout.txt");
  char kind[32], name[64];
  while (fscanf(f, "%31s", kind) == 1) {
    if (!strcmp(kind, "scalar")) {
      int v; fscanf(f, "%63s %d", name, &v);
      if (!strcmp(name, "d_model")) d_model = v;
      else if (!strcmp(name, "n_head")) n_head = v;
      else if (!strcmp(name, "n_layer")) n_layer = v;
      else if (!strcmp(name, "d_ff")) d_ff = v;
      else if (!strcmp(name, "block_size")) block_size = v;
      else if (!strcmp(name, "vocab_size")) vocab_size = v;
      else if (!strcmp(name, "new_tokens")) new_tokens = v;
      else if (!strcmp(name, "total_floats")) total_floats = v;
    } else if (!strcmp(kind, "prompt")) {
      fscanf(f, "%d", &prompt_len);
      for (int i = 0; i < prompt_len; i++) fscanf(f, "%d", &prompt[i]);
    } else if (!strcmp(kind, "tensor")) {
      Tensor *t = &tensors[n_tensors++];
      fscanf(f, "%63s %d %d", t->name, &t->offset, &t->count);
    }
  }
  fclose(f);
  head_dim = d_model / n_head;
}

static void load_weights(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) die("cannot open weights.bin");
  W = malloc((size_t)total_floats * sizeof(float));
  if (!W) die("oom");
  if (fread(W, sizeof(float), total_floats, f) != (size_t)total_floats) die("short read");
  fclose(f);
}

static void load_vocab(const char *path) {
  FILE *f = fopen(path, "r");
  if (!f) die("cannot open vocab.txt");
  for (int i = 0; i < vocab_size; i++) fscanf(f, "%d", &vocab_cp[i]);
  fclose(f);
}

/* ---------------------------------------------------------------- the one difference */
/* out[T,N] = x[T,K] @ w[K,N] + bias[N] */
static void matmul(float *out, const float *x, const float *w, const float *bias,
                   int Tn, int K, int N) {
#if USE_ACCELERATE
  for (int t = 0; t < Tn; t++) memcpy(out + (size_t)t * N, bias, (size_t)N * sizeof(float));
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Tn, N, K,
              1.0f, x, K, w, N, 1.0f, out, N);
#else
  for (int t = 0; t < Tn; t++) {
    const float *xr = x + (size_t)t * K;
    float *orow = out + (size_t)t * N;
    for (int j = 0; j < N; j++) {
      float acc = bias[j];
      for (int k = 0; k < K; k++) acc += xr[k] * w[(size_t)k * N + j];
      orow[j] = acc;
    }
  }
#endif
}
/* ------------------------------------------------------------------------------------ */

static void layernorm(float *out, const float *x, const float *g, const float *b,
                      int Tn, int n) {
  for (int t = 0; t < Tn; t++) {
    const float *r = x + (size_t)t * n;
    float *o = out + (size_t)t * n;
    float m = 0.f;
    for (int j = 0; j < n; j++) m += r[j];
    m /= n;
    float v = 0.f;
    for (int j = 0; j < n; j++) { float z = r[j] - m; v += z * z; }
    v /= n;
    float inv = 1.0f / sqrtf(v + 1e-5f);
    for (int j = 0; j < n; j++) o[j] = (r[j] - m) * inv * g[j] + b[j];
  }
}

static float gelu(float z) {
  const float S = 0.7978845608028654f; /* sqrt(2/pi) */
  return 0.5f * z * (1.0f + tanhf(S * (z + 0.044715f * z * z * z)));
}

static float *buf_x, *buf_h, *buf_qkv, *buf_o, *buf_p, *buf_f1, *buf_f2, *logits;

static void forward(const int *ids, int Tn) {
  const float *tok = T("tok.weight"), *pos = T("pos.weight");
  for (int t = 0; t < Tn; t++)
    for (int j = 0; j < d_model; j++)
      buf_x[(size_t)t * d_model + j] =
          tok[(size_t)ids[t] * d_model + j] + pos[(size_t)t * d_model + j];

  char nm[64];
  float scale = 1.0f / sqrtf((float)head_dim);
  for (int L = 0; L < n_layer; L++) {
#define TB(field) (sprintf(nm, "b%d." field, L), T(nm))
    layernorm(buf_h, buf_x, TB("ln1.g"), TB("ln1.b"), Tn, d_model);
    matmul(buf_qkv, buf_h, TB("qkv.w"), TB("qkv.b"), Tn, d_model, 3 * d_model);

    memset(buf_o, 0, (size_t)Tn * d_model * sizeof(float));
    for (int h = 0; h < n_head; h++) {
      int base = h * head_dim;
      for (int p = 0; p < Tn; p++) {
        const float *q = buf_qkv + (size_t)p * 3 * d_model + base;
        float sc[MAXT], mx = -1e30f;
        for (int u = 0; u <= p; u++) {
          const float *k = buf_qkv + (size_t)u * 3 * d_model + d_model + base;
          float s = 0.f;
          for (int j = 0; j < head_dim; j++) s += q[j] * k[j];
          s *= scale;
          sc[u] = s;
          if (s > mx) mx = s;
        }
        float tot = 0.f;
        for (int u = 0; u <= p; u++) { sc[u] = expf(sc[u] - mx); tot += sc[u]; }
        float *o = buf_o + (size_t)p * d_model + base;
        for (int u = 0; u <= p; u++) {
          float wgt = sc[u] / tot;
          const float *v = buf_qkv + (size_t)u * 3 * d_model + 2 * d_model + base;
          for (int j = 0; j < head_dim; j++) o[j] += wgt * v[j];
        }
      }
    }
    matmul(buf_p, buf_o, TB("proj.w"), TB("proj.b"), Tn, d_model, d_model);
    for (size_t i = 0; i < (size_t)Tn * d_model; i++) buf_x[i] += buf_p[i];

    layernorm(buf_h, buf_x, TB("ln2.g"), TB("ln2.b"), Tn, d_model);
    matmul(buf_f1, buf_h, TB("fc1.w"), TB("fc1.b"), Tn, d_model, d_ff);
    for (size_t i = 0; i < (size_t)Tn * d_ff; i++) buf_f1[i] = gelu(buf_f1[i]);
    matmul(buf_f2, buf_f1, TB("fc2.w"), TB("fc2.b"), Tn, d_ff, d_model);
    for (size_t i = 0; i < (size_t)Tn * d_model; i++) buf_x[i] += buf_f2[i];
#undef TB
  }
  layernorm(buf_h, buf_x, T("lnf.g"), T("lnf.b"), Tn, d_model);
  matmul(logits, buf_h + (size_t)(Tn - 1) * d_model, T("head.w"), T("head.b"),
         1, d_model, vocab_size);
}

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
  const char *dir = (argc > 1) ? argv[1] : "../model";
  char p[512];
  snprintf(p, sizeof p, "%s/layout.txt", dir);  load_layout(p);
  snprintf(p, sizeof p, "%s/weights.bin", dir); load_weights(p);
  snprintf(p, sizeof p, "%s/vocab.txt", dir);   load_vocab(p);

  buf_x  = malloc((size_t)MAXT * d_model * sizeof(float));
  buf_h  = malloc((size_t)MAXT * d_model * sizeof(float));
  buf_qkv= malloc((size_t)MAXT * 3 * d_model * sizeof(float));
  buf_o  = malloc((size_t)MAXT * d_model * sizeof(float));
  buf_p  = malloc((size_t)MAXT * d_model * sizeof(float));
  buf_f1 = malloc((size_t)MAXT * d_ff * sizeof(float));
  buf_f2 = malloc((size_t)MAXT * d_model * sizeof(float));
  logits = malloc((size_t)vocab_size * sizeof(float));

  int ids[MAXT + 512];
  int n = prompt_len;
  memcpy(ids, prompt, (size_t)prompt_len * sizeof(int));

  int live = isatty(2);
  double t0 = now();
  for (int i = 0; i < new_tokens; i++) {
    int start = (n > block_size) ? n - block_size : 0;
    int Tn = n - start;
    forward(ids + start, Tn);
    int best = 0;
    for (int j = 1; j < vocab_size; j++) if (logits[j] > logits[best]) best = j;
    ids[n++] = best;
    if (live) {
      double el = now() - t0;
      fprintf(stderr, "\r  %-16s [%-30.*s] %3d/%d  %7.2fs  %8.3f tok/s", IMPL_NAME,
              (int)(30.0 * (i + 1) / new_tokens),
              "##############################", i + 1, new_tokens, el, (i + 1) / el);
      fflush(stderr);
    }
  }
  double el = now() - t0;
  if (live) fprintf(stderr, "\n");

  double lsum = 0.0;
  for (int j = 0; j < vocab_size; j++) lsum += logits[j];

  printf("{\"impl\":\"%s\",\"tokens\":[", IMPL_NAME);
  for (int i = 0; i < n; i++) printf("%s%d", i ? "," : "", ids[i]);
  printf("],\"text\":\"");
  for (int i = 0; i < n; i++) {
    int cp = vocab_cp[ids[i]];
    if (cp == '\n') printf("\\n");
    else if (cp == '"') printf("\\\"");
    else if (cp == '\\') printf("\\\\");
    else printf("%c", cp);
  }
  printf("\",\"elapsed_s\":%.6f,\"tokens_per_s\":%.4f,\"logit_sum\":%.4f,"
         "\"forward_passes\":%d}\n", el, new_tokens / el, lsum, new_tokens);
  return 0;
}
