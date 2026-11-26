// qwen-oracle: ik_llama.cpp reference for PicoLM divergence hunting.
//
// Three modes of the same executable:
//
//   QORACLE_FAST=1  fast reference path, NO eval callback. Prints
//                   "gen[N] argmax=<id>" for each greedy step, so long
//                   sequences can be compared token-for-token.
//   default         same, plus a per-position logit dump and (optionally)
//                   top-k. See QORACLE_LOGITS / QORACLE_TOPK.
//   QORACLE_STATS   installs the ggml eval callback and prints one line per
//                   selected intermediate tensor in PicoLM's "STAT" format.
//
// Token ids come from QORACLE_TOKENS (comma separated) so the prompt is
// byte-for-byte identical to what PicoLM tokenized; otherwise QORACLE_PROMPT
// is tokenized with add_bos from the model metadata.
//
//   QORACLE_LOGITS=/path/a.bin   raw f32 logits per position (pos-major)
//   QORACLE_TOPK=N               print top-N logits of every position
//   QORACLE_STATS=substr,...     tensor-name substrings to dump ("all")
//   QORACLE_POS=0                only dump logits at these positions
//   QORACLE_N=32                 tokens to generate (greedy)

#include "common.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct dump_state {
    std::vector<std::string> pats;
    bool all = false;
    int  n_dumped = 0;
    int  n_budget = 4000;
};

static bool name_selected(const char * name, const dump_state & st) {
    if (st.all) return true;
    for (const auto & p : st.pats) if (strstr(name, p.c_str())) return true;
    return false;
}

static void dump_tensor(struct ggml_tensor * t, const dump_state & st) {
    if (!t || !t->name || !t->name[0]) return;
    if (!name_selected(t->name, st)) return;
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16) return;
    const int64_t n = ggml_nelements(t);
    if (n <= 0 || n > (1 << 24)) return;

    std::vector<float> buf((size_t) n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, buf.data(), 0, (size_t) n * sizeof(float));
    } else {
        std::vector<ggml_fp16_t> h((size_t) n);
        ggml_backend_tensor_get(t, h.data(), 0, (size_t) n * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(h.data(), buf.data(), n);
    }

    double ss = 0.0, crc = 0.0;
    float mn = buf[0], mx = buf[0], amax = fabsf(buf[0]);
    int n_nan = 0, n_inf = 0;
    for (int64_t i = 0; i < n; i++) {
        float x = buf[(size_t) i];
        if (std::isnan(x)) { n_nan++; continue; }
        if (std::isinf(x)) { n_inf++; continue; }
        ss += (double) x * x;
        crc += (double) x * (double) (i + 1);
        if (x < mn) mn = x;
        if (x > mx) mx = x;
        if (fabsf(x) > amax) amax = fabsf(x);
    }
    const double rms = sqrt(ss / (double) n);
    printf("GLM5 STAT %s n=%lld rms=%.6f min=%.6f max=%.6f absmax=%.6f crc=%.9e nan=%d inf=%d v:",
           t->name, (long long) n, rms, mn, mx, amax, crc, n_nan, n_inf);
    for (int64_t i = 0; i < 4 && i < n; i++) printf(" %.6f", buf[(size_t) i]);
    printf("\n");
    fflush(stdout);

    /* QORACLE_DUMP=<dir>: write raw f32 vectors for element-wise comparison */
    const char * dumpdir = getenv("QORACLE_DUMP");
    if (dumpdir && dumpdir[0]) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s.f32", dumpdir, t->name);
        FILE * fp = fopen(path, "wb");
        if (fp) { fwrite(buf.data(), sizeof(float), (size_t) n, fp); fclose(fp); }
    }
}

static int eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    dump_state * st = (dump_state *) user_data;
    if (ask) {
        if (t && t->name && name_selected(t->name, *st) && st->n_dumped < st->n_budget) return true;
        return false;
    }
    dump_tensor(t, *st);
    st->n_dumped++;
    return true;
}

static std::vector<int> parse_ints(const char * s) {
    std::vector<int> v;
    if (!s) return v;
    std::string str(s);
    size_t pos = 0;
    while (pos < str.size()) {
        size_t comma = str.find(',', pos);
        if (comma == std::string::npos) comma = str.size();
        std::string tok = str.substr(pos, comma - pos);
        if (!tok.empty()) v.push_back(atoi(tok.c_str()));
        pos = comma + 1;
    }
    return v;
}

int main(int argc, char ** argv) {
    gpt_params params;
    params.n_predict = 1;
    params.prompt = "O";
    params.n_ctx = 512;

    if (!gpt_params_parse(argc, argv, params)) {
        fprintf(stderr, "usage: %s -m model.gguf [-n N] [-c N]\n", argv[0]);
        return 1;
    }

    const char * env_stats = getenv("QORACLE_STATS");
    const bool want_stats = env_stats != nullptr && strcmp(env_stats, "none") != 0;
    const char * env_logits = getenv("QORACLE_LOGITS");
    const int topk = getenv("QORACLE_TOPK") ? atoi(getenv("QORACLE_TOPK")) : 0;
    std::vector<int> only_pos = parse_ints(getenv("QORACLE_POS"));
    const int n_gen = getenv("QORACLE_N") ? atoi(getenv("QORACLE_N")) : (params.n_predict > 0 ? params.n_predict : 1);

    dump_state st;
    if (want_stats) {
        std::string s(env_stats);
        if (s == "all") st.all = true;
        else {
            size_t pos = 0;
            while (pos <= s.size()) {
                size_t comma = s.find(',', pos);
                if (comma == std::string::npos) comma = s.size();
                if (comma > pos) st.pats.push_back(s.substr(pos, comma - pos));
                pos = comma + 1;
            }
        }
    }

    llama_backend_init();

    llama_model_params mparams = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.c_str(), mparams);
    if (!model) { fprintf(stderr, "error: cannot load model\n"); return 1; }

    llama_context_params cparams = common_context_params_to_llama(params);
    if (want_stats) {
        cparams.cb_eval = eval_cb;
        cparams.cb_eval_user_data = &st;
    }
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "error: cannot create context\n"); return 1; }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    std::vector<llama_token> toks;
    if (const char * tk = getenv("QORACLE_TOKENS")) {
        for (int t : parse_ints(tk)) toks.push_back((llama_token) t);
    } else {
        toks = common_tokenize(ctx, params.prompt, true);
    }
    if (toks.empty()) { fprintf(stderr, "error: no tokens\n"); return 1; }

    FILE * lf = nullptr;
    if (env_logits) {
        lf = fopen(env_logits, "wb");
        if (!lf) { fprintf(stderr, "error: cannot open %s\n", env_logits); return 1; }
    }

    auto emit_logits = [&](int pos) {
        if (!lf && topk <= 0) return;
        bool want = only_pos.empty();
        for (int p : only_pos) if (p == pos) want = true;
        if (!want) return;
        const float * logits = llama_get_logits_ith(ctx, -1);
        if (lf) {
            fwrite(&pos, sizeof(int), 1, lf);
            fwrite(&n_vocab, sizeof(int), 1, lf);
            fwrite(logits, sizeof(float), (size_t) n_vocab, lf);
            fflush(lf);
        }
        if (topk > 0) {
            std::vector<int> idx((size_t) n_vocab);
            for (int i = 0; i < n_vocab; i++) idx[i] = i;
            std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(),
                              [&](int a, int b) { return logits[a] > logits[b]; });
            fprintf(stderr, "[TOPLOG pos=%d]", pos);
            for (int k = 0; k < topk; k++)
                fprintf(stderr, " %d=%.4f", idx[k], logits[idx[k]]);
            fprintf(stderr, "\n");
        }
    };

    llama_batch batch = llama_batch_get_one(toks.data(), (int) toks.size(), 0, 0);
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "error: decode failed\n"); return 1; }

    int pos = (int) toks.size() - 1;
    emit_logits(pos);

    for (int i = 0; i < n_gen; i++) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        int best = 0;
        for (int v = 1; v < n_vocab; v++) if (logits[v] > logits[best]) best = v;
        printf("gen[%d] argmax=%d\n", i, best);
        fflush(stdout);
        if (i + 1 == n_gen) break;
        llama_batch b1 = llama_batch_get_one((llama_token *) &best, 1, (llama_pos)(toks.size() + i), 0);
        if (llama_decode(ctx, b1) != 0) { fprintf(stderr, "error: decode failed\n"); return 1; }
        pos++;
        emit_logits(pos);
    }

    if (lf) fclose(lf);
    fprintf(stderr, "qwen-oracle: %zu prompt tokens, %d generated, %d tensors dumped\n",
            toks.size(), n_gen, st.n_dumped);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return 0;
}
