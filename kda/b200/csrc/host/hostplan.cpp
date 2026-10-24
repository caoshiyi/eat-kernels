// C++ port of the host planners of the CAKE KDA kernels (base/kernel.py: plan / _lpt / _refine_bins / split_tail / wrap_plan / schedule / plain_pieces,
// kernel.py: cp_plan and helpers, fix_sched.fix_schedule, base/dual.py lpt_lists).  Line-by-line mirror of the Python (same float operation order, same
// tie-breaking, incl. CPython's set pop order in _dual_pieces), so every plan is IDENTICAL to the Python planner's (tests: test_hostplan.py).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <queue>
#include <set>
#include <tuple>
#include <vector>

namespace py = pybind11;
using std::vector;

namespace {
// ---------------------------------------------------------------- base/kernel.py constants
constexpr int CIO_MIN_CHUNKS = 17;
constexpr double CIO_BAR = 1.10, CIO_BOUNDARY_SAVING = 0.63, CIO_CHUNK_OVERHEAD = 0.0;
constexpr double M64_CHUNK_COST = 0.93, TASK_COST_PERSIST = 1.0, TASK_COST_LAUNCH = 8.0;
constexpr double PLAIN_TASK = 3.4, PLAIN_SHORT = -0.5;
constexpr double PPC_EPS = 1.10, PPC_PIECE = 5.6;
constexpr int PACK_MAX_TASKS = 2500;
constexpr double SPLIT_BOUNDARY = 4.0;
constexpr int PRELOAD_MIN = 17;
constexpr double CIO_EPS = 1.0, CIO_ENTRY = 2.77, CIO_FRONT = 0.48, CIO_BACK = 0.08, CIO_SHORT = 0.70, CIO_BACK_BLOCK = 2.4, CIO_SHORT_STORE = 0.2;

struct Entry { int t, c0, c1, si, so; };
using List = vector<Entry>;

// CPython >= 3.12 float sum(): Neumaier-compensated, starting from int 0 (the first float item is added exactly).  Empty sum -> 0.
struct PySum {
    double f = 0.0, c = 0.0; bool first = true;
    void add(double x) {
        if (first) { f = x; first = false; return; }
        const double t = f + x;
        if (std::fabs(f) >= std::fabs(x)) c += (f - t) + x; else c += (x - t) + f;
        f = t;
    }
    double result() const { if (first) return 0.0; double r = f; if (c != 0.0 && std::isfinite(c)) r += c; return r; }
};

inline int cdiv32(int64_t L) { return (int)((L + 31) / 32); }   // (L + 31) // 32 (L >= 0)

double plain_task(int n) { return n + PLAIN_TASK + (n < 17 ? PLAIN_SHORT : 0.0); }

// ---------------------------------------------------------------- heaps
// min-heap of (load, bin) pairs, lexicographic (the Python heapq of tuples)
using LB = std::pair<double, int>;
struct LBGreater { bool operator()(const LB& a, const LB& b) const { return a.first > b.first || (a.first == b.first && a.second > b.second); } };

double makespan(const vector<double>& costs, int n_sms) {   // _makespan: greedy list scheduling in launch order
    int n = std::min<int>(n_sms, (int)costs.size());
    std::priority_queue<double, vector<double>, std::greater<double>> free_;
    for (int i = 0; i < n; i++) free_.push(0.0);
    double end = 0.0;
    for (double c : costs) {
        double t = free_.top(); free_.pop(); t += c;
        end = std::max(end, t);
        free_.push(t);
    }
    return end;
}

double lpt_makespan(vector<double> costs, int n) {   // kernel.py _lpt_makespan
    std::sort(costs.begin(), costs.end(), std::greater<double>());
    std::priority_queue<double, vector<double>, std::greater<double>> heap;
    for (int i = 0; i < std::max(1, n); i++) heap.push(0.0);
    double mx = 0.0;
    for (double c : costs) { double t = heap.top(); heap.pop(); heap.push(t + c); }
    while (!heap.empty()) { mx = std::max(mx, heap.top()); heap.pop(); }
    return mx;
}

// _lpt: returns (makespan, bins)
std::pair<double, vector<vector<int>>> lpt(const vector<double>& costs, int n_bins) {
    vector<int> order(costs.size());
    for (size_t i = 0; i < order.size(); i++) order[i] = (int)i;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return costs[a] > costs[b]; });
    std::priority_queue<LB, vector<LB>, LBGreater> heap;
    for (int b = 0; b < n_bins; b++) heap.push({0.0, b});
    vector<vector<int>> bins(n_bins);
    for (int i : order) {
        LB t = heap.top(); heap.pop();
        bins[t.second].push_back(i);
        heap.push({t.first + costs[i], t.second});
    }
    double mx = 0.0; bool first = true;
    while (!heap.empty()) { double l = heap.top().first; heap.pop(); if (first || l > mx) mx = l; first = false; }
    return {mx, bins};
}

double sum_costs(const vector<double>& costs, const vector<int>& b) {
    PySum s;
    for (int i : b) s.add(costs[i]);
    return s.result();
}

vector<vector<int>> refine_bins(const vector<double>& costs, vector<vector<int>> bins, int max_iter = 2000) {
    const int nb = (int)bins.size();
    vector<double> loads(nb);
    for (int i = 0; i < nb; i++) loads[i] = sum_costs(costs, bins[i]);
    for (int it = 0; it < max_iter; it++) {
        int k = 0;
        for (int i = 1; i < nb; i++) if (loads[i] > loads[k]) k = i;       // max(): first maximal index
        const double mk = loads[k];
        bool have = false; double bc = 0; int ba = -1, bj = -1, bb = -1;
        for (int j = 0; j < nb; j++) {
            if (j == k) continue;
            for (int a : bins[k]) {
                for (int b : bins[j]) {
                    const double d = costs[a] - costs[b];
                    if (d > 1e-9 && loads[j] + d < mk - 1e-9) {
                        const double cand = std::max(loads[j] + d, loads[k] - d);
                        if (!have || cand < bc) { have = true; bc = cand; ba = a; bj = j; bb = b; }
                    }
                }
            }
        }
        if (!have) break;
        auto rm = [](vector<int>& v, int x) { v.erase(std::find(v.begin(), v.end(), x)); };
        rm(bins[k], ba); rm(bins[bj], bb); bins[k].push_back(bb); bins[bj].push_back(ba);
        const double d = costs[ba] - costs[bb]; loads[k] -= d; loads[bj] += d;
    }
    vector<vector<int>> out;
    for (auto& b : bins) {
        vector<int> s = b;
        std::stable_sort(s.begin(), s.end(), [&](int x, int y) { return costs[x] > costs[y]; });
        out.push_back(std::move(s));
    }
    return out;
}

struct Plan {   // plan(): variant 64 -> no sched; 128 -> (task_list, cta_ptr)
    int variant;
    vector<int> order;
    bool has_sched = false;
    vector<int> task_list, cta_ptr;
};

Plan plan(const vector<int64_t>& seq_lens, int H, int n_sms, bool allow_m64) {
    const int N = (int)seq_lens.size();
    vector<int> chunks(N);
    for (int i = 0; i < N; i++) chunks[i] = cdiv32(seq_lens[i]);
    const int ntask = N * H;
    vector<double> costs(ntask), pcosts(ntask);
    for (int n = 0; n < N; n++) for (int h = 0; h < H; h++) { costs[n * H + h] = chunks[n] + TASK_COST_PERSIST; pcosts[n * H + h] = plain_task(chunks[n]); }
    const int n_cta = std::max(1, std::min(n_sms, ntask));
    const double m128 = lpt(costs, n_cta).first;
    auto bins = lpt(pcosts, n_cta).second;
    if (ntask <= PACK_MAX_TASKS) bins = refine_bins(pcosts, bins);
    Plan p;
    p.order.resize(N);
    for (int i = 0; i < N; i++) p.order[i] = i;
    std::stable_sort(p.order.begin(), p.order.end(), [&](int a, int b) { return seq_lens[a] > seq_lens[b]; });
    vector<double> m64_tasks;
    m64_tasks.reserve((size_t)ntask * 2);
    for (int i : p.order) for (int r = 0; r < 2 * H; r++) m64_tasks.push_back(chunks[i] * M64_CHUNK_COST + TASK_COST_LAUNCH);
    const double m64 = makespan(m64_tasks, n_sms);
    if (allow_m64 && m64 < m128) { p.variant = 64; return p; }
    p.variant = 128; p.has_sched = true;
    p.cta_ptr.push_back(0);
    for (auto& b : bins) {
        for (int i : b) p.task_list.push_back(i);   // tasks[i] = (i // H, i % H): n*H + h == i
        p.cta_ptr.push_back((int)p.task_list.size());
    }
    return p;
}

// ---------------------------------------------------------------- cio list costs
double entry_cost(int n, bool front, bool back, bool shrt) {
    return n * CIO_EPS + CIO_ENTRY + (front ? CIO_FRONT : 0.0) + (back ? CIO_BACK : 0.0) + (shrt ? CIO_SHORT : 0.0);
}

double list_cost(const List& lst) {
    double c = 0.0; int prev = -1; bool has_prev = false;
    const size_t m = lst.size();
    for (size_t i = 0; i < m; i++) {
        const Entry& e = lst[i];
        const int n = e.c1 - e.c0;
        c += entry_cost(n, e.so >= 0, e.si >= 0, i + 1 < m && n < PRELOAD_MIN);
        if (e.si >= 0 && (!has_prev || prev < PRELOAD_MIN)) c += CIO_BACK_BLOCK - CIO_BACK;
        if (has_prev && n < PRELOAD_MIN) c += CIO_SHORT_STORE;
        prev = n; has_prev = true;
    }
    return c;
}

double max_list_cost(const vector<List>& ls) {
    double m = 0.0; bool first = true;
    for (auto& l : ls) { double c = list_cost(l); if (first || c > m) m = c; first = false; }
    return m;
}

// split_tail: (lists, nslot)
std::pair<vector<List>, int> split_tail(const vector<int>& task_chunks, const vector<vector<int>>& bins, int n_cta, double boundary = SPLIT_BOUNDARY) {
    const int nt = (int)task_chunks.size();
    vector<double> cost(nt);
    for (int i = 0; i < nt; i++) cost[i] = task_chunks[i] + 1.0;
    vector<vector<int>> whole = bins;
    struct P4 { int t, c0, c1, slot; };
    vector<vector<P4>> front(n_cta), back(n_cta);
    vector<double> load(n_cta);
    for (int i = 0; i < n_cta; i++) load[i] = sum_costs(cost, whole[i]);
    int nslot = 0;
    for (int iter = 0; iter < 4 * n_cta; iter++) {
        int k = 0;
        for (int i = 1; i < n_cta; i++) if (load[i] > load[k]) k = i;
        if (whole[k].empty()) break;
        vector<int> cand;
        for (int i : whole[k]) if (task_chunks[i] >= PRELOAD_MIN) cand.push_back(i);
        if (cand.empty()) break;
        int t = cand[0];
        for (int i : cand) if (cost[i] < cost[t]) t = i;     // min(): first minimal
        const int n = task_chunks[t];
        vector<int> order;
        for (int i = 0; i < n_cta; i++) if (i != k) order.push_back(i);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return load[a] < load[b]; });
        if (order.size() < 2) break;
        const int x = std::min(order[0], order[1]), y = std::max(order[0], order[1]);
        // s = int(round((load[y] - load[x] + n) / 2.0))
        int s = (int)std::nearbyint((load[y] - load[x] + n) / 2.0);
        s = std::max(1, std::min(n - 1, s));
        const double new_x = load[x] + s + 1.0 + boundary, new_y = load[y] + (n - s) + 1.0 + boundary;
        if (std::max(std::max(new_x, new_y), load[k] - cost[t]) >= load[k] - 0.5) break;
        PySum ae;
        for (auto& f : front[x]) ae.add(f.c1 - f.c0 + boundary);
        const double a_end = ae.result() + s + boundary + 17;
        if (load[y] < a_end + boundary) break;
        whole[k].erase(std::find(whole[k].begin(), whole[k].end(), t));
        load[k] -= cost[t];
        front[x].push_back({t, 0, s, nslot});
        back[y].push_back({t, s, n, nslot});
        load[x] = new_x; load[y] = new_y;
        nslot++;
    }
    double mx0 = 0.0; bool first = true;
    for (auto& b : bins) { double s = sum_costs(cost, b); if (first || s > mx0) mx0 = s; first = false; }
    double mxl = 0.0; first = true;
    for (int i = 0; i < n_cta; i++) { if (first || load[i] > mxl) mxl = load[i]; first = false; }
    if (mxl > mx0 - 1.0) {
        front.assign(n_cta, {}); back.assign(n_cta, {});
        whole = bins; nslot = 0;
    }
    vector<List> lists;
    for (int i = 0; i < n_cta; i++) {
        List lst;
        for (auto& f : front[i]) lst.push_back({f.t, f.c0, f.c1, -1, f.slot});
        for (int t : whole[i]) lst.push_back({t, 0, task_chunks[t], -1, -1});
        for (auto& b : back[i]) lst.push_back({b.t, b.c0, b.c1, b.slot, -1});
        lists.push_back(std::move(lst));
    }
    return {lists, nslot};
}

// wrap_plan: nullopt if no capacity is feasible
bool wrap_plan(const vector<int>& task_chunks, int n_cta, vector<List>& out, int& nslot_out) {
    const int nt = (int)task_chunks.size();
    vector<int> order(nt);
    for (int i = 0; i < nt; i++) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return task_chunks[a] > task_chunks[b]; });
    PySum tot;
    for (int n : task_chunks) tot.add(entry_cost(n, false, false, false));
    const double total = tot.result();
    const int lo = (int)(total / n_cta);
    for (int L4 = 4 * lo; L4 < 4 * lo + 256; L4++) {
        const double L = L4 / 4.0;
        vector<List> lists(n_cta), heads(n_cta);
        vector<double> head_cost(n_cta, 0.0);
        int j = n_cta - 1, nslot = 0; double load = 0.0; bool ok = true;
        for (int t : order) {
            const int n = task_chunks[t];
            while (true) {
                if (j < 0) { ok = false; break; }
                const double w = entry_cost(n, false, false, 0 < n && n < PRELOAD_MIN);
                if (load + w <= L) { lists[j].push_back({t, 0, n, -1, -1}); load += w; break; }
                // prev entry before the back piece on CTA j
                bool has_prev = false; int prev_len = 0;
                if (!lists[j].empty()) { has_prev = true; prev_len = lists[j].back().c1 - lists[j].back().c0; }
                else if (!heads[j].empty()) { has_prev = true; prev_len = heads[j].back().c1 - heads[j].back().c0; }
                const double bb = (has_prev && prev_len >= PRELOAD_MIN) ? CIO_BACK : CIO_BACK_BLOCK;
                const int a = (int)((L - load - CIO_ENTRY - bb) / CIO_EPS);   // Python int(): truncation toward zero
                const int b = n - a;
                const double fcost = entry_cost(b, true, false, b < PRELOAD_MIN);
                if (n >= PRELOAD_MIN && a >= 1 && b >= 1 && j >= 1 && fcost + 16 + 8 <= L - (entry_cost(a, false, false, false) + bb)) {
                    lists[j].push_back({t, b, n, nslot, -1});
                    heads[j - 1].push_back({t, 0, b, -1, nslot});
                    head_cost[j - 1] = fcost;
                    nslot++; j--; load = fcost;
                    break;
                }
                j--;
                load = j >= 0 ? head_cost[j] : 0.0;
            }
            if (!ok) break;
        }
        if (!ok) continue;
        out.assign(n_cta, {});
        for (int i = 0; i < n_cta; i++) {
            List backs, wholes;
            for (auto& e : lists[i]) (e.si >= 0 ? backs : wholes).push_back(e);
            List o = heads[i];
            o.insert(o.end(), wholes.begin(), wholes.end());
            o.insert(o.end(), backs.begin(), backs.end());
            out[i] = std::move(o);
        }
        nslot_out = nslot;
        return true;
    }
    return false;
}

struct CioPlan { vector<int> ids; vector<List> lists; int nslot; };
struct Schedule {
    int variant;
    vector<int> order;
    bool has_sched = false;
    vector<int> task_list, cta_ptr;
    bool use_cio = false;
    bool has_cio = false;
    CioPlan cio;
};

vector<int> sorted_unique(const vector<int>& v) { vector<int> s = v; std::sort(s.begin(), s.end()); s.erase(std::unique(s.begin(), s.end()), s.end()); return s; }

Schedule schedule(const vector<int64_t>& seq_lens, int H, int n_sms) {
    Plan pl = plan(seq_lens, H, n_sms, true);
    Schedule S; S.variant = pl.variant; S.order = pl.order; S.has_sched = pl.has_sched; S.task_list = pl.task_list; S.cta_ptr = pl.cta_ptr;
    const int N = (int)seq_lens.size();
    auto chunks_of = [&](int t) { return cdiv32(seq_lens[t / H]); };
    bool use_cio = false;
    int minc = 1 << 30;
    for (int i = 0; i < N; i++) minc = std::min(minc, cdiv32(seq_lens[i]));
    if (S.has_sched && minc >= CIO_MIN_CHUNKS) {
        std::pair<int64_t, int> best{-1, -1}; bool first = true;
        for (size_t i = 0; i + 1 < S.cta_ptr.size(); i++) {
            int64_t s = 0; int cnt = 0;
            for (int p = S.cta_ptr[i]; p < S.cta_ptr[i + 1]; p++) { s += chunks_of(S.task_list[p]); cnt++; }
            std::pair<int64_t, int> v{s, cnt};
            if (first || v > best) best = v;
            first = false;
        }
        const int busy_tasks = best.second; const int64_t busy_chunks = best.first;
        use_cio = (busy_tasks - 1) * CIO_BOUNDARY_SAVING > CIO_CHUNK_OVERHEAD * (double)busy_chunks;
    }
    if (S.has_sched && !use_cio) {
        const int n_cta0 = (int)S.cta_ptr.size() - 1;
        vector<vector<int>> bins0(n_cta0);
        for (int i = 0; i < n_cta0; i++) bins0[i].assign(S.task_list.begin() + S.cta_ptr[i], S.task_list.begin() + S.cta_ptr[i + 1]);
        vector<int> ids0 = sorted_unique(S.task_list);
        std::map<int, int> pos0; for (size_t i = 0; i < ids0.size(); i++) pos0[ids0[i]] = (int)i;
        vector<int> tch0; for (int t : ids0) tch0.push_back(chunks_of(t));
        double whole0 = 0.0; bool first = true;
        for (auto& b : bins0) { PySum ps; for (int t : b) ps.add(plain_task(tch0[pos0[t]])); const double s = ps.result(); if (first || s > whole0) whole0 = s; first = false; }
        vector<vector<int>> pb(n_cta0);
        for (int i = 0; i < n_cta0; i++) for (int t : bins0[i]) pb[i].push_back(pos0[t]);
        auto st = split_tail(tch0, pb, n_cta0);
        const double sp0 = max_list_cost(st.first);
        if (st.second > 0 && sp0 * CIO_BAR < whole0) use_cio = true;
    }
    bool wrapped = false; vector<int> w_ids; vector<List> w_lists; int w_nslot = 0;
    if (S.has_sched) {
        const int n_ctw = (int)S.cta_ptr.size() - 1;
        vector<int> idw = sorted_unique(S.task_list);
        vector<int> tchw; for (int t : idw) tchw.push_back(chunks_of(t));
        vector<List> wl; int wns = 0;
        if (wrap_plan(tchw, n_ctw, wl, wns)) {
            const double wcost = max_list_cost(wl);
            double base_cost;
            std::map<int, int> posw; for (size_t i = 0; i < idw.size(); i++) posw[idw[i]] = (int)i;
            if (use_cio) {
                vector<vector<int>> bw(n_ctw);
                for (int i = 0; i < n_ctw; i++) for (int p = S.cta_ptr[i]; p < S.cta_ptr[i + 1]; p++) bw[i].push_back(posw[S.task_list[p]]);
                auto ls = split_tail(tchw, bw, n_ctw);
                base_cost = max_list_cost(ls.first);
            } else {
                base_cost = 0.0; bool first = true;
                for (int i = 0; i < n_ctw; i++) {
                    PySum ps;
                    for (int p = S.cta_ptr[i]; p < S.cta_ptr[i + 1]; p++) ps.add(plain_task(tchw[posw[S.task_list[p]]]));
                    const double s = ps.result();
                    if (first || s > base_cost) base_cost = s;
                    first = false;
                }
            }
            if (wcost * (use_cio ? 1.02 : CIO_BAR) < base_cost) { wrapped = true; w_ids = idw; w_lists = wl; w_nslot = wns; use_cio = true; }
        }
    }
    S.use_cio = use_cio;
    if (use_cio) {
        S.has_cio = true;
        if (wrapped) { S.cio = {w_ids, w_lists, w_nslot}; }
        else {
            const int n_cta = (int)S.cta_ptr.size() - 1;
            vector<int> ids = sorted_unique(S.task_list);
            std::map<int, int> pos; for (size_t i = 0; i < ids.size(); i++) pos[ids[i]] = (int)i;
            vector<int> tchunks; for (int t : ids) tchunks.push_back(chunks_of(t));
            vector<vector<int>> bt(n_cta);
            for (int i = 0; i < n_cta; i++) for (int p = S.cta_ptr[i]; p < S.cta_ptr[i + 1]; p++) bt[i].push_back(pos[S.task_list[p]]);
            auto sl = split_tail(tchunks, bt, n_cta);
            S.cio = {ids, sl.first, sl.second};
        }
    }
    return S;
}

// plain_pieces: nullopt -> keep whole tasks
bool plain_pieces(const vector<int64_t>& seq_lens, int H, const vector<int>& task_list, const vector<int>& cta_ptr, CioPlan& out) {
    const int n_ct = (int)cta_ptr.size() - 1;
    vector<int> ids = sorted_unique(task_list);
    std::map<int, int> pos; for (size_t i = 0; i < ids.size(); i++) pos[ids[i]] = (int)i;
    vector<int> tch; for (int t : ids) tch.push_back(cdiv32(seq_lens[t / H]));
    vector<vector<int>> bins(n_ct);
    for (int i = 0; i < n_ct; i++) for (int p = cta_ptr[i]; p < cta_ptr[i + 1]; p++) bins[i].push_back(pos[task_list[p]]);
    double whole = 0.0; bool first = true;
    for (auto& b : bins) { PySum ps; for (int t : b) ps.add(plain_task(tch[t])); const double s = ps.result(); if (first || s > whole) whole = s; first = false; }
    auto st = split_tail(tch, bins, n_ct);
    if (st.second == 0) return false;
    double cost = 0.0; first = true;
    for (auto& l : st.first) {
        PySum ps;
        for (auto& e : l) ps.add((e.c1 - e.c0) * PPC_EPS + PLAIN_TASK + ((e.si >= 0 || e.so >= 0) ? PPC_PIECE : 0.0));
        const double s = ps.result();
        if (first || s > cost) cost = s;
        first = false;
    }
    if (cost * 1.02 < whole) { out = {ids, st.first, st.second}; return true; }
    return false;
}

// ---------------------------------------------------------------- kernel.py: CP planner
constexpr double STEP_US = 1.70, TASK_COST = 1.0, CONTENTION = 1.0, FIX_BW = 4.5e6, FIX_STATE_US = 0.08, FIX_FIXED_US = 6.0;
constexpr int SHORT_CHUNKS = 17;
constexpr double SHORT_TASK_COST = 5.0, MIN_GAIN = 0.04, DUAL_F = 1.40, DUAL_TASK = 6.4;
constexpr double MULTI_MARGIN = 0.05, SNAP_STEPS = 3.0, FIX_ITEM_US = 6.0, FIX_CHAIN_US = 0.07;
constexpr int PIECE_TASKS_PER_CTA = 3, PIECE_MIN = 8;
constexpr int64_t SNAP_BYTES_MAX = (int64_t)3 << 30;
constexpr double INF = std::numeric_limits<double>::infinity();

// CPython 3.12 set of small non-negative ints (setobject.c): emulates the table layout so that pop() order is identical.
struct PySet {
    static constexpr int LINEAR_PROBES = 9, PERTURB_SHIFT = 5;
    struct Ent { int64_t key; bool used = false; bool dummy = false; };
    vector<Ent> table; size_t mask = 7; size_t fill = 0, used = 0; size_t finger = 0;
    PySet() { table.assign(8, Ent{}); }
    static void insert_clean(vector<Ent>& tab, size_t mask, int64_t key) {
        size_t i = (size_t)key & mask; size_t perturb = (size_t)key;
        while (true) {
            size_t probes = (i + LINEAR_PROBES <= mask) ? LINEAR_PROBES : 0;
            size_t e = i;
            while (true) {
                if (!tab[e].used) { tab[e].key = key; tab[e].used = true; tab[e].dummy = false; return; }
                if (probes == 0) break;
                probes--; e++;
            }
            perturb >>= PERTURB_SHIFT;
            i = (i * 5 + 1 + perturb) & mask;
        }
    }
    void resize(size_t minused) {
        size_t newsize = 8;
        while (newsize <= minused) newsize <<= 1;
        vector<Ent> nt(newsize);
        const size_t nm = newsize - 1;
        for (size_t i = 0; i < table.size(); i++) if (table[i].used && !table[i].dummy) insert_clean(nt, nm, table[i].key);
        table = std::move(nt); mask = nm; fill = used;
    }
    void add(int64_t key) {
        size_t i = (size_t)key & mask; size_t perturb = (size_t)key;
        int64_t freeslot = -1;
        while (true) {
            size_t probes = (i + LINEAR_PROBES <= mask) ? LINEAR_PROBES : 0;
            size_t e = i;
            while (true) {
                Ent& en = table[e];
                if (!en.used) {   // unused slot: insert (into the first dummy seen, if any)
                    if (freeslot < 0) {
                        fill++; used++; en.key = key; en.used = true; en.dummy = false;
                        if (fill * 5 < mask * 3) return;
                        resize(used > 50000 ? used * 2 : used * 4);
                        return;
                    }
                    Ent& f = table[freeslot]; used++; f.key = key; f.used = true; f.dummy = false; return;
                }
                if (!en.dummy && en.key == key) return;
                if (en.dummy && freeslot < 0) freeslot = (int64_t)e;
                if (probes == 0) break;
                probes--; e++;
            }
            perturb >>= PERTURB_SHIFT;
            i = (i * 5 + 1 + perturb) & mask;
        }
    }
    void merge(const PySet& other) {   // set_merge
        if (other.used == 0) return;
        if ((fill + other.used) * 5 >= mask * 3) resize((used + other.used) * 2);
        if (fill == 0 && mask == other.mask && other.fill == other.used) {
            for (size_t i = 0; i <= other.mask; i++) if (other.table[i].used && !other.table[i].dummy) { table[i] = other.table[i]; }
            fill = other.fill; used = other.used; return;
        }
        if (fill == 0) {
            fill = other.used; used = other.used;
            for (size_t i = 0; i <= other.mask; i++) if (other.table[i].used && !other.table[i].dummy) insert_clean(table, mask, other.table[i].key);
            return;
        }
        for (size_t i = 0; i <= other.mask; i++) if (other.table[i].used && !other.table[i].dummy) add(other.table[i].key);
    }
    bool empty() const { return used == 0; }
    int64_t pop() {
        size_t e = finger & mask;
        while (!table[e].used || table[e].dummy) { e++; if (e > mask) e = 0; }
        const int64_t key = table[e].key;
        table[e].dummy = true; used--;
        finger = e + 1;
        return key;
    }
};

// set(range(lo, hi + 1, step)) | {hi}
PySet make_cands(int lo, int hi, int step) {
    PySet a;
    for (int v = lo; v <= hi; v += step) a.add(v);
    PySet copy;      // set_copy: set_merge into an empty new set
    copy.merge(a);
    PySet one; one.add(hi);
    copy.merge(one);
    return copy;
}

struct DualBest { bool has = false; double mk = 0; int s1 = 0, s2 = 0; };

DualBest dual_pieces(int tcx, int H, int n_D) {
    DualBest best;
    if (n_D < 2 || tcx < 2 * PIECE_MIN || 2 * H > PIECE_TASKS_PER_CTA * n_D) return best;
    const int lo = PIECE_MIN, hi = tcx / 2;
    const int step = std::max(4, ((hi - lo) / 10) / 4 * 4);
    PySet cands = make_cands(lo, hi, step);
    std::set<int> done;
    auto mkspan = [&](int s1, int s2) {
        vector<double> c;
        c.insert(c.end(), H, DUAL_F * s1 + DUAL_TASK);
        c.insert(c.end(), H, DUAL_F * s2 + DUAL_TASK);
        return lpt_makespan(c, n_D);
    };
    while (!cands.empty()) {
        int s2 = (int)cands.pop();
        if (done.count(s2) || s2 < lo || s2 > hi) continue;
        done.insert(s2);
        int s1 = tcx - s2;
        if (s1 % 4) {
            s1 -= s1 % 4;
            s2 = tcx - s1;
            if (s2 > s1 || done.count(s2)) continue;
            done.insert(s2);
        }
        const double mk = mkspan(s1, s2);
        if (!best.has || mk < best.mk) { best.has = true; best.mk = mk; best.s1 = s1; best.s2 = s2; }
    }
    if (best.has && step > 4) {
        const int s2c = best.s2;
        for (int s2 = std::max(lo, s2c - step); s2 <= std::min(hi, s2c + step); s2 += 4) {
            if (done.count(s2)) continue;
            done.insert(s2);
            const int s1 = tcx - s2;
            if (s1 % 4 || s2 > s1) continue;
            const double mk = mkspan(s1, s2);
            if (mk < best.mk) { best.mk = mk; best.s1 = s1; best.s2 = s2; }
        }
    }
    return best;
}

struct GC { std::map<std::tuple<vector<int64_t>, int, int>, double> memo; };

double group_cost(GC& gc, const vector<int64_t>& seq_lens, int H, int n_sms) {
    auto key = std::make_tuple(seq_lens, H, n_sms);
    auto it = gc.memo.find(key);
    if (it != gc.memo.end()) return it->second;
    const int N = (int)seq_lens.size();
    vector<int> chunks(N);
    for (int i = 0; i < N; i++) chunks[i] = cdiv32(seq_lens[i]);
    Schedule S = schedule(seq_lens, H, n_sms);
    double c;
    if (S.has_cio) c = max_list_cost(S.cio.lists);
    else if (S.variant == 64) {
        vector<double> v; for (int i : S.order) for (int r = 0; r < 2 * H; r++) v.push_back(chunks[i] * M64_CHUNK_COST + TASK_COST_LAUNCH);
        c = makespan(v, n_sms);
    } else {
        vector<double> v; for (int n = 0; n < N; n++) for (int h = 0; h < H; h++) v.push_back(chunks[n] + TASK_COST);
        c = lpt_makespan(v, n_sms);
    }
    bool any_short = false;
    for (int ch : chunks) if (0 < ch && ch < SHORT_CHUNKS) any_short = true;
    if (any_short) {
        vector<double> v;
        for (int ch : chunks) if (ch > 0) for (int h = 0; h < H; h++) v.push_back(ch + (ch < SHORT_CHUNKS ? SHORT_TASK_COST : TASK_COST));
        c = std::max(c, lpt_makespan(v, n_sms));
    }
    gc.memo[key] = c;
    return c;
}

double direct_cost(GC& gc, const vector<int64_t>& seq_lens, int H, int n_sms) {
    const int N = (int)seq_lens.size();
    vector<int> chunks(N);
    for (int i = 0; i < N; i++) chunks[i] = cdiv32(seq_lens[i]);
    const double lp = group_cost(gc, seq_lens, H, n_sms);
    PySum ps; int mx = 0;
    for (int c : chunks) { ps.add(c + TASK_COST); mx = std::max(mx, c); }
    const double bound = std::max(mx + TASK_COST, ps.result() * H / n_sms);
    return std::min(lp, bound);
}

double fix_us(int64_t tail, int H) { return FIX_FIXED_US + (double)(tail * H * 128 * 2 * 3) / FIX_BW + FIX_STATE_US * H; }

std::pair<int, int> split_budget(double w1, double w2, int n) {
    if (w2 <= 0) return {n, 0};
    if (w1 <= 0) return {0, n};
    int n2 = std::min(n - 1, std::max(1, (int)std::nearbyint(n * w2 / (w1 + w2))));
    return {n - n2, n2};
}

std::pair<int, int> even_split(int n_p, int n_s) {
    if (n_s <= 0 || n_p % 2 == 0) return {n_p, n_s};
    const int tot = n_p + n_s;
    if (n_p >= 3) n_p -= 1; else n_p += 1;
    return {n_p, tot - n_p};
}

double side_cost(GC& gc, const vector<int64_t>& seq_lens, int H, int n_sms) {
    if ((int64_t)seq_lens.size() * H <= 4LL * n_sms) return group_cost(gc, seq_lens, H, n_sms);
    vector<double> tc;
    for (int64_t L : seq_lens) { int c = cdiv32(L); if (c > 0) tc.push_back(c + (c < SHORT_CHUNKS ? SHORT_TASK_COST : TASK_COST)); }
    if (tc.empty()) return 0.0;
    double mx = tc[0]; PySum ps;
    for (double v : tc) { mx = std::max(mx, v); ps.add(v); }
    return std::max(mx, ps.result() * H / n_sms);
}

struct CpOpt { bool ok = false; double cost = 0; int n_P = 0, n_S = 0, n_B = 0; bool has_pieces = false; int p1 = 0, p2 = 0; bool p_first = false; };

}  // namespace

// cp_plan(seq_lens, H, n_sms, ck_n) -> None | (j, x, n_P, n_S, n_B, pieces | None, p_first)
py::object cp_plan(const vector<int64_t>& seq_lens, int H, int n_sms, int ck_n) {
    GC gc;
    const int N = (int)seq_lens.size();
    vector<int> chunks(N);
    for (int i = 0; i < N; i++) chunks[i] = cdiv32(seq_lens[i]);
    auto tcost = [&](int ch) { return ch > 0 ? ch + (ch < SHORT_CHUNKS ? SHORT_TASK_COST : TASK_COST) : 0.0; };
    {
        PySum ps; int mx = 0;
        for (int c : chunks) { ps.add(tcost(c)); mx = std::max(mx, c); }
        if (mx + TASK_COST <= (1.0 + MIN_GAIN) * ps.result() * H / n_sms) return py::none();
    }
    const double direct = direct_cost(gc, seq_lens, H, n_sms);
    bool have_best = false; double best_cost = 0; int best_j = 0, best_x = 0; CpOpt best_opt;
    vector<int> jorder(N);
    for (int i = 0; i < N; i++) jorder[i] = i;
    std::stable_sort(jorder.begin(), jorder.end(), [&](int a, int b) { return seq_lens[a] > seq_lens[b]; });
    for (int jj = 0; jj < std::min(2, N); jj++) {
        const int j = jorder[jj];
        const int64_t T = seq_lens[j];
        if (T < 32 * 8) continue;
        vector<int64_t> pre_lens(seq_lens.begin(), seq_lens.begin() + j), suf_lens(seq_lens.begin() + j + 1, seq_lens.end());
        PySum psuf, ppre;
        for (int n = j + 1; n < N; n++) psuf.add(tcost(chunks[n]));
        const double w_suf = psuf.result() * H;
        for (int n = 0; n < j; n++) ppre.add(tcost(chunks[n]));
        const double w_pre = ppre.result() * H;
        const int tc = cdiv32(T);
        auto cost_at = [&](int xc, int n_A, bool multi) -> CpOpt {
            CpOpt r;
            int n_B = (n_sms - n_A) / 2;
            if (n_B > 1) n_B -= n_B % 2;
            auto sb = split_budget(w_pre + (xc + TASK_COST) * H, w_suf, n_A);
            auto es = even_split(sb.first, sb.second);
            const int n_P = es.first, n_S = es.second;
            if (n_P < 1 || (w_suf > 0 && n_S < 1)) return r;
            vector<int64_t> pl = pre_lens; pl.push_back(32LL * xc);
            const double a = std::max(side_cost(gc, pl, H, n_P), w_suf > 0 ? side_cost(gc, suf_lens, H, n_S) : 0.0);
            const double b0 = group_cost(gc, vector<int64_t>{T - 32LL * xc}, H, n_B);
            const int n_D = n_sms - n_A;
            const int rows = ck_n ? std::max(0, (tc - 1) / ck_n - xc / ck_n) : 0;
            if ((int64_t)rows * H * 2 * 65536 > SNAP_BYTES_MAX) return r;
            const double snap = rows * SNAP_STEPS;
            const double b_d = lpt_makespan(vector<double>(H, DUAL_F * (tc - xc) + DUAL_TASK + snap), n_D);
            const double fix = (fix_us(T - 32LL * xc, H) + ((H + n_sms - 1) / n_sms) * rows * FIX_ITEM_US) / STEP_US;
            const double b = ck_n ? INF : b0;
            CpOpt opt; opt.ok = true;
            if (b_d < b) { opt.cost = CONTENTION * std::max(a, b_d) + fix; opt.n_P = n_P; opt.n_S = n_S; opt.n_B = -n_D; opt.p_first = a >= b_d; }
            else { opt.cost = CONTENTION * std::max(a, b) + fix; opt.n_P = n_P; opt.n_S = n_S; opt.n_B = n_B; opt.p_first = a >= b; }
            if (multi) {
                DualBest mp = dual_pieces(tc - xc, H, n_D);
                if (mp.has) {
                    const double cm = std::max(a, CONTENTION * (mp.mk + snap)) + fix + FIX_CHAIN_US * H / STEP_US;
                    if (cm < opt.cost * (1.0 - MULTI_MARGIN)) {
                        opt.cost = cm; opt.n_P = n_P; opt.n_S = n_S; opt.n_B = -n_D; opt.has_pieces = true; opt.p1 = mp.s1; opt.p2 = mp.s2;
                        opt.p_first = a >= CONTENTION * (mp.mk + snap);
                    }
                }
            }
            return opt;
        };
        std::set<int> nas = {H, n_sms / 2, (2 * n_sms) / 3, H + (n_sms - H) / 4, H + (n_sms - H) / 3, n_sms - H, n_sms - (H + 1) / 2};
        for (int n_A : nas) {   // std::set iterates ascending = sorted(set(...))
            if (n_A > 2) n_A -= n_A % 2;
            if (n_A < 1 || (n_sms - n_A) / 2 < 1) continue;
            const int step = std::max(4, (tc - 7) / 24);
            vector<std::pair<CpOpt, int>> coarse;
            for (int xc = 4; xc < tc - 3; xc += step) { CpOpt c = cost_at(xc, n_A, true); if (c.ok) coarse.push_back({c, xc}); }
            if (coarse.empty()) continue;
            size_t bi = 0;
            for (size_t i = 1; i < coarse.size(); i++) if (coarse[i].first.cost < coarse[bi].first.cost) bi = i;
            const int xc0 = coarse[bi].second;
            for (int xc = std::max(4, xc0 - step + 1); xc < std::min(tc - 3, xc0 + step); xc++) {
                CpOpt r = cost_at(xc, n_A, true);
                if (r.ok && (!have_best || r.cost < best_cost)) { have_best = true; best_cost = r.cost; best_j = j; best_x = 32 * xc; best_opt = r; }
            }
        }
    }
    if (!have_best || best_cost > direct * (1.0 - MIN_GAIN)) return py::none();
    py::object pieces = best_opt.has_pieces ? py::object(py::make_tuple(best_opt.p1, best_opt.p2)) : py::object(py::none());
    return py::make_tuple(best_j, best_x, best_opt.n_P, best_opt.n_S, best_opt.n_B, pieces, py::bool_(best_opt.p_first));
}

// ---- Python-visible views of the base planner (equality tests / table building)
py::object lists_to_py(const vector<List>& ls) {
    py::list out;
    for (auto& l : ls) {
        py::list o;
        for (auto& e : l) o.append(py::make_tuple(e.t, e.c0, e.c1, e.si, e.so));
        out.append(o);
    }
    return out;
}

py::tuple schedule_py(const vector<int64_t>& seq_lens, int H, int n_sms) {
    Schedule S = schedule(seq_lens, H, n_sms);
    py::object sched = py::none();
    if (S.has_sched) sched = py::make_tuple(S.task_list, S.cta_ptr);
    py::object cio = py::none();
    if (S.has_cio) cio = py::make_tuple(S.cio.ids, lists_to_py(S.cio.lists), S.cio.nslot);
    return py::make_tuple(S.variant, S.order, sched, S.use_cio, cio);
}

py::object plain_pieces_py(const vector<int64_t>& seq_lens, int H, const vector<int>& task_list, const vector<int>& cta_ptr) {
    CioPlan o;
    if (!plain_pieces(seq_lens, H, task_list, cta_ptr, o)) return py::none();
    return py::make_tuple(o.ids, lists_to_py(o.lists), o.nslot);
}

py::object dual_pieces_py(int tcx, int H, int n_D) {
    DualBest b = dual_pieces(tcx, H, n_D);
    if (!b.has) return py::none();
    return py::make_tuple(b.mk, py::make_tuple(b.s1, b.s2));
}

vector<int> pyset_pop_order(int lo, int hi, int step) {   // test hook: the pop order of set(range(lo, hi+1, step)) | {hi}
    PySet s = make_cands(lo, hi, step);
    vector<int> o;
    while (!s.empty()) o.push_back((int)s.pop());
    return o;
}

py::object lpt_py(const vector<double>& costs, int n) { auto r = lpt(costs, n); return py::make_tuple(r.first, r.second); }
vector<vector<int>> refine_py(const vector<double>& costs, const vector<vector<int>>& bins) { return refine_bins(costs, bins); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("lpt", &lpt_py);
    m.def("refine", &refine_py);
    m.def("cp_plan", &cp_plan, py::arg("seq_lens"), py::arg("H"), py::arg("n_sms"), py::arg("ck_n") = 0);
    m.def("schedule", &schedule_py);
    m.def("plain_pieces", &plain_pieces_py);
    m.def("dual_pieces", &dual_pieces_py);
    m.def("pyset_pop_order", &pyset_pop_order);
}
